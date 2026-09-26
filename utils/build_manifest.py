import os
import re
import zipfile
import pandas as pd
import pydicom
import io


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "lumos_clinical_data.xlsx"
)

XRAY_ZIP = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "lumos_x_001_280_dcm.zip"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "xray_manifest.csv"
)


# ---------------------------------------------------------
# Load clinical data
# ---------------------------------------------------------

print("Loading clinical data...")

clinical = pd.read_excel(CLINICAL_FILE)

clinical["patient_id"] = clinical["patient_id"].astype(int)

print(f"Clinical patients: {len(clinical)}")


# ---------------------------------------------------------
# Read X-ray DICOM metadata
# ---------------------------------------------------------

print("Reading X-ray DICOM metadata...")

records = []

with zipfile.ZipFile(XRAY_ZIP, "r") as z:

    dicom_files = [
        name
        for name in z.namelist()
        if name.lower().endswith(".dcm")
        and not name.startswith("__MACOSX")
    ]

    print(f"DICOM files found: {len(dicom_files)}")

    for name in dicom_files:

        match = re.search(r"lumos_x_(\d+)", name)

        if not match:
            continue

        patient_id = int(match.group(1))

        data = z.read(name)

        ds = pydicom.dcmread(
            io.BytesIO(data),
            stop_before_pixels=True
        )

        # Use SeriesDescription only for view classification,
        # matching the validated X-ray audit logic.
        series_description = str(
            ds.get("SeriesDescription", "")
        ).lower()

        # Determine view
        if (
            "lateral" in series_description
            or "lat" in series_description
            or "侧位" in series_description
        ):
            view = "Lateral"

        elif (
            "ap" in series_description
            or "前后位" in series_description
        ):
            view = "AP"

        elif "extension" in series_description:
            view = "Extension"

        else:
            view = "Unknown"

        records.append({
            "patient_id": patient_id,
            "dicom_file": name,
            "view": view,
            "series_description": str(
                ds.get("SeriesDescription", "")
            ),
            "instance_number": ds.get(
                "InstanceNumber", ""
            ),
            "rows": ds.get("Rows", ""),
            "columns": ds.get("Columns", "")
        })


xray = pd.DataFrame(records)


# ---------------------------------------------------------
# Count views per patient
# ---------------------------------------------------------

view_table = (
    xray
    .groupby(["patient_id", "view"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for column in ["AP", "Lateral", "Extension", "Unknown"]:
    if column not in view_table.columns:
        view_table[column] = 0


# ---------------------------------------------------------
# Create patient-level X-ray manifest
# ---------------------------------------------------------

manifest = clinical[
    [
        "patient_id",
        "age",
        "gender",
        "BMI",
        "bmd",
        "t_value",
        "Osteoporosis"
    ]
].copy()

manifest = manifest.merge(
    view_table,
    on="patient_id",
    how="left"
)

for column in ["AP", "Lateral", "Extension", "Unknown"]:
    manifest[column] = manifest[column].fillna(0).astype(int)


# ---------------------------------------------------------
# Add training eligibility
# ---------------------------------------------------------

manifest["has_ap"] = manifest["AP"] > 0
manifest["has_lateral"] = manifest["Lateral"] > 0

manifest["strict_xray_pair"] = (
    manifest["has_ap"]
    & manifest["has_lateral"]
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("========================================")
print("MANIFEST CREATED")
print("========================================")
print(f"Patients: {len(manifest)}")
print(f"Patients with AP: {manifest['has_ap'].sum()}")
print(f"Patients with Lateral: {manifest['has_lateral'].sum()}")
print(
    f"Patients with AP + Lateral: "
    f"{manifest['strict_xray_pair'].sum()}"
)

print()
print("Label distribution:")
print(
    manifest["Osteoporosis"]
    .value_counts()
    .sort_index()
)

print()
print("X-ray view counts:")
print(
    manifest[["AP", "Lateral", "Extension", "Unknown"]]
    .sum()
)

print()
print("Saved:")
print(OUTPUT_FILE)