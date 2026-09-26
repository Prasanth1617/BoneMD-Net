import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


MASTER_FILE = os.path.join(
    RESULTS_DIR,
    "multimodal_master_manifest.csv"
)

XRAY_FILE = os.path.join(
    RESULTS_DIR,
    "xray_image_mapping.csv"
)

CT_MANIFEST_FILES = [
    os.path.join(RESULTS_DIR, "ct_manifest_001_070.csv"),
    os.path.join(RESULTS_DIR, "ct_manifest_071_140.csv"),
    os.path.join(RESULTS_DIR, "ct_manifest_141_210.csv"),
    os.path.join(RESULTS_DIR, "ct_manifest_211_280.csv"),
]

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "final_multimodal_manifest.csv"
)


def normalize_columns(df):
    """Normalize known column-name variations."""
    rename_map = {
        "Osteoporosis": "label",
        "ct_slices": "dicom_count",
        "AP": "has_ap",
        "Lateral": "has_lateral",
    }

    return df.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in df.columns
        }
    )


def choose_xray(group, view):
    """
    Deterministically choose one X-ray for a patient/view.
    Priority:
      1. lowest instance number
      2. series UID
      3. filename
    """
    candidates = group[group["view"] == view].copy()

    if candidates.empty:
        return None

    candidates["instance_sort"] = pd.to_numeric(
        candidates["instance_number"],
        errors="coerce"
    )

    candidates["instance_sort"] = candidates["instance_sort"].fillna(
        float("inf")
    )

    candidates = candidates.sort_values(
        by=[
            "instance_sort",
            "series_instance_uid",
            "dicom_file",
        ],
        na_position="last"
    )

    return candidates.iloc[0]["dicom_file"]


print("=" * 70)
print("BoneMD-Net Final Multimodal Manifest Builder")
print("=" * 70)


# -------------------------------------------------------------------
# 1. Load master multimodal manifest
# -------------------------------------------------------------------

print("\n[1/5] Loading multimodal master manifest...")

master = pd.read_csv(MASTER_FILE)
master = normalize_columns(master)

print(f"Master rows: {len(master)}")
print(f"Master columns: {list(master.columns)}")


required_master = [
    "patient_id",
    "label",
    "strict_xray_pair",
    "multimodal_eligible",
]

missing = [
    col for col in required_master
    if col not in master.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in master manifest: {missing}"
    )


# -------------------------------------------------------------------
# 2. Load X-ray mapping
# -------------------------------------------------------------------

print("\n[2/5] Loading verified X-ray mapping...")

xray = pd.read_csv(XRAY_FILE)

required_xray = [
    "patient_id",
    "dicom_file",
    "view",
    "series_instance_uid",
    "instance_number",
]

missing = [
    col for col in required_xray
    if col not in xray.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in X-ray mapping: {missing}"
    )

xray["patient_id"] = pd.to_numeric(
    xray["patient_id"],
    errors="raise"
).astype(int)

print(f"X-ray records: {len(xray)}")
print(f"X-ray patients: {xray['patient_id'].nunique()}")


# -------------------------------------------------------------------
# 3. Select deterministic AP + Lateral image
# -------------------------------------------------------------------

print("\n[3/5] Selecting AP and Lateral images...")

eligible = master[
    master["multimodal_eligible"].astype(bool)
].copy()

eligible["patient_id"] = pd.to_numeric(
    eligible["patient_id"],
    errors="raise"
).astype(int)

selected_rows = []

for _, row in eligible.iterrows():

    patient_id = int(row["patient_id"])

    patient_xray = xray[
        xray["patient_id"] == patient_id
    ]

    ap_file = choose_xray(
        patient_xray,
        "AP"
    )

    lateral_file = choose_xray(
        patient_xray,
        "Lateral"
    )

    selected_rows.append({
        "patient_id": patient_id,
        "ap_dicom_file": ap_file,
        "lateral_dicom_file": lateral_file,
    })


selected_xray = pd.DataFrame(selected_rows)

print(
    f"Eligible patients: {len(selected_xray)}"
)

print(
    "Missing AP:",
    selected_xray["ap_dicom_file"].isna().sum()
)

print(
    "Missing Lateral:",
    selected_xray["lateral_dicom_file"].isna().sum()
)


# -------------------------------------------------------------------
# 4. Load CT manifests and verify CT coverage
# -------------------------------------------------------------------

print("\n[4/5] Loading CT manifests...")

ct_frames = []

for filename in CT_MANIFEST_FILES:

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"CT manifest not found: {filename}"
        )

    df = pd.read_csv(filename)
    df = normalize_columns(df)

    required = [
        "patient_id",
        "dicom_count",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{os.path.basename(filename)} "
            f"is missing columns: {missing}"
        )

    df["patient_id"] = pd.to_numeric(
        df["patient_id"],
        errors="raise"
    ).astype(int)

    df["ct_manifest_source"] = os.path.basename(
        filename
    )

    ct_frames.append(df)


ct = pd.concat(
    ct_frames,
    ignore_index=True
)

print(f"Total CT patients: {len(ct)}")
print(
    f"Unique CT patients: "
    f"{ct['patient_id'].nunique()}"
)

if ct["patient_id"].duplicated().any():
    duplicates = ct[
        ct["patient_id"].duplicated(keep=False)
    ]["patient_id"].tolist()

    raise ValueError(
        f"Duplicate CT patient IDs found: {duplicates}"
    )


# -------------------------------------------------------------------
# Derive CT ZIP from patient number
# -------------------------------------------------------------------

def get_ct_zip(patient_id):

    if 1 <= patient_id <= 70:
        return "lumos_ct_001_070_dcm.zip"

    if 71 <= patient_id <= 140:
        return "lumos_ct_071_140_dcm.zip"

    if 141 <= patient_id <= 210:
        return "lumos_ct_141_210_dcm.zip"

    if 211 <= patient_id <= 280:
        return "lumos_ct_211_280_dcm.zip"

    return None


def get_ct_folder(patient_id):
    return f"lumos_ct_{patient_id:03d}"


ct["ct_zip"] = ct["patient_id"].apply(
    get_ct_zip
)

ct["ct_patient_folder"] = ct["patient_id"].apply(
    get_ct_folder
)


# -------------------------------------------------------------------
# 5. Merge everything
# -------------------------------------------------------------------

print("\n[5/5] Building final manifest...")

final = eligible.merge(
    selected_xray,
    on="patient_id",
    how="left",
    validate="one_to_one",
)

final = final.merge(
    ct[
        [
            "patient_id",
            "dicom_count",
            "ct_zip",
            "ct_patient_folder",
        ]
    ],
    on="patient_id",
    how="left",
    validate="one_to_one",
)


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    f"Final rows: {len(final)}"
)

print(
    f"Unique patients: "
    f"{final['patient_id'].nunique()}"
)

print(
    f"Patient range: "
    f"{final['patient_id'].min()} - "
    f"{final['patient_id'].max()}"
)

print(
    f"Duplicate patients: "
    f"{final['patient_id'].duplicated().sum()}"
)


required_final = [
    "patient_id",
    "label",
    "ap_dicom_file",
    "lateral_dicom_file",
    "dicom_count_x",
    "dicom_count_y",
    "ct_zip",
    "ct_patient_folder",
]

missing_counts = final[
    required_final
].isna().sum()

print("\nMissing values:")
print(missing_counts)


print("\nLabel distribution:")
print(
    final["label"]
    .value_counts()
    .sort_index()
)


# -------------------------------------------------------------------
# Hard validation checks
# -------------------------------------------------------------------

assert len(final) == 272, (
    f"Expected 272 eligible patients, "
    f"found {len(final)}"
)

assert final["patient_id"].nunique() == 272

assert final["patient_id"].duplicated().sum() == 0

assert final["label"].notna().all()

assert final["ap_dicom_file"].notna().all()

assert final["lateral_dicom_file"].notna().all()

assert final["dicom_count_x"].notna().all()
assert final["dicom_count_y"].notna().all()

assert final["ct_zip"].notna().all()

assert final["ct_patient_folder"].notna().all()


expected_labels = {
    0: 122,
    1: 79,
    2: 71,
}

actual_labels = (
    final["label"]
    .value_counts()
    .sort_index()
    .to_dict()
)

assert actual_labels == expected_labels, (
    f"Unexpected labels: {actual_labels}"
)


# -------------------------------------------------------------------
# Select useful columns
# -------------------------------------------------------------------

final = final.rename(
    columns={
        "dicom_count_x": "xray_dicom_count",
        "dicom_count_y": "ct_dicom_count",
    }
)

preferred_columns = [
    "patient_id",
    "label",
    "age",
    "gender",
    "BMI",
    "bmd",
    "t_value",
    "ap_dicom_file",
    "lateral_dicom_file",
    "ct_zip",
    "ct_patient_folder",
    "xray_dicom_count",
    "ct_dicom_count",
    "strict_xray_pair",
    "multimodal_eligible",
]

available_columns = [
    col for col in preferred_columns
    if col in final.columns
]

final = final[available_columns]


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Patients: {len(final)}"
)

print(
    "Labels:",
    final["label"].value_counts().sort_index().to_dict()
)

print("\nFirst 5 rows:")
print(final.head().to_string(index=False))
