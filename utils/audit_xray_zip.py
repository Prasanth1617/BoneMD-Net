import zipfile
import io
import re
from pathlib import Path

import pandas as pd
import pydicom


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

CLINICAL_FILE = DATASET_DIR / "lumos_clinical_data.xlsx"
XRAY_ZIP = DATASET_DIR / "lumos_x_001_280_dcm.zip"
OUTPUT_FILE = RESULTS_DIR / "xray_audit.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_patient_number(filename):
    """
    Extract patient number from paths such as:
    lumos_x_001/lumos_x_001_1.Dcm
    """
    match = re.search(r"lumos_x_(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


def normalize_view(series_description):
    """
    Convert DICOM series description into:
    AP / Lateral / Extension / Unknown
    """
    if not series_description:
        return "Unknown"

    text = str(series_description).lower()

    if (
        "lateral" in text
        or "lat" in text
        or "侧位" in text
    ):
        return "Lateral"

    if (
        "ap" in text
        or "前后位" in text
    ):
        return "AP"

    if "extension" in text:
        return "Extension"

    return "Unknown"


# ------------------------------------------------------------
# Main audit
# ------------------------------------------------------------
def main():

    print("=" * 70)
    print("BoneMD-Net LUMOS X-ray ZIP Audit")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------
    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Clinical file not found:\n{CLINICAL_FILE}"
        )

    if not XRAY_ZIP.exists():
        raise FileNotFoundError(
            f"X-ray ZIP not found:\n{XRAY_ZIP}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load clinical data
    # --------------------------------------------------------
    print("\nLoading clinical data...")

    clinical = pd.read_excel(CLINICAL_FILE)

    print(f"Clinical patients: {len(clinical)}")
    print(f"Clinical columns: {len(clinical.columns)}")

    # Create lookup by patient ID
    clinical_lookup = clinical.set_index("patient_id")

    # --------------------------------------------------------
    # Inspect ZIP
    # --------------------------------------------------------
    print("\nOpening X-ray ZIP...")
    print(XRAY_ZIP)

    records = []

    with zipfile.ZipFile(XRAY_ZIP, "r") as z:

        all_files = z.namelist()

        dicom_files = [
            f
            for f in all_files
            if f.lower().endswith(".dcm")
            and not f.startswith("__MACOSX")
        ]

        print(f"\nTotal ZIP entries: {len(all_files)}")
        print(f"Actual DICOM files: {len(dicom_files)}")

        # ----------------------------------------------------
        # Read metadata from every DICOM
        # ----------------------------------------------------
        for index, filename in enumerate(dicom_files, start=1):

            patient_number = get_patient_number(filename)

            if patient_number is None:
                print(
                    f"WARNING: Could not determine patient number: "
                    f"{filename}"
                )
                continue

            try:
                data = z.read(filename)

                ds = pydicom.dcmread(
                    io.BytesIO(data),
                    stop_before_pixels=True
                )

                series_description = ds.get(
                    "SeriesDescription",
                    ""
                )

                view = normalize_view(series_description)

                # ------------------------------------------------
                # Clinical information
                # ------------------------------------------------
                if patient_number in clinical_lookup.index:

                    patient = clinical_lookup.loc[patient_number]

                    label = patient.get(
                        "Osteoporosis",
                        None
                    )

                    age = patient.get(
                        "age",
                        None
                    )

                    gender = patient.get(
                        "gender",
                        None
                    )

                    bmd = patient.get(
                        "bmd",
                        None
                    )

                    t_value = patient.get(
                        "t_value",
                        None
                    )

                else:

                    label = None
                    age = None
                    gender = None
                    bmd = None
                    t_value = None

                records.append(
                    {
                        "patient_id": patient_number,
                        "dicom_file": filename,
                        "view": view,
                        "series_description": str(
                            series_description
                        ),
                        "study_instance_uid": str(
                            ds.get(
                                "StudyInstanceUID",
                                ""
                            )
                        ),
                        "series_instance_uid": str(
                            ds.get(
                                "SeriesInstanceUID",
                                ""
                            )
                        ),
                        "instance_number": ds.get(
                            "InstanceNumber",
                            None
                        ),
                        "rows": ds.get(
                            "Rows",
                            None
                        ),
                        "columns": ds.get(
                            "Columns",
                            None
                        ),
                        "photometric_interpretation": str(
                            ds.get(
                                "PhotometricInterpretation",
                                ""
                            )
                        ),
                        "patient_id_dicom": str(
                            ds.get(
                                "PatientID",
                                ""
                            )
                        ),
                        "label": label,
                        "age": age,
                        "gender": gender,
                        "bmd": bmd,
                        "t_value": t_value,
                    }
                )

            except Exception as e:

                print(
                    f"\nERROR reading {filename}: {e}"
                )

            # Progress every 50 files
            if index % 50 == 0:
                print(
                    f"Processed {index}/{len(dicom_files)} DICOM files..."
                )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------
    audit = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    audit.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nDICOM files processed: {len(audit)}")

    print(
        f"Unique patient folders: "
        f"{audit['patient_id'].nunique()}"
    )

    print("\nView distribution:")
    print(
        audit["view"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDICOM PatientID values:")
    print(
        "Blank:",
        (
            audit["patient_id_dicom"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
    )

    print("\nClinical labels represented:")
    print(
        audit["label"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nDICOMs per patient:")
    print(
        audit.groupby("patient_id")
        .size()
        .describe()
        .to_string()
    )

    print("\nPatients with both AP and Lateral:")
    views_per_patient = (
        audit.groupby("patient_id")["view"]
        .apply(set)
    )

    both = views_per_patient.apply(
        lambda x: "AP" in x and "Lateral" in x
    )

    print(
        f"{both.sum()} / {len(both)} patients"
    )

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nFirst 20 audit records:")
    print(
        audit.head(20).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()