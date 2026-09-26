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

XRAY_ZIP = DATASET_DIR / "lumos_x_001_280_dcm.zip"
OUTPUT_FILE = RESULTS_DIR / "xray_image_mapping.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_patient_number(filename):
    match = re.search(r"lumos_x_(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


def normalize_view(series_description):
    """
    Exact same classification logic used by audit_xray_zip.py.
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
# Main
# ------------------------------------------------------------
def main():

    print("=" * 70)
    print("BoneMD-Net LUMOS X-ray Image Mapping")
    print("=" * 70)

    if not XRAY_ZIP.exists():
        raise FileNotFoundError(
            f"X-ray ZIP not found:\n{XRAY_ZIP}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    print("\nOpening X-ray ZIP...")
    print(XRAY_ZIP)

    with zipfile.ZipFile(XRAY_ZIP, "r") as z:

        all_files = z.namelist()

        dicom_files = [
            f
            for f in all_files
            if f.lower().endswith(".dcm")
            and not f.startswith("__MACOSX")
        ]

        print(f"Total ZIP entries: {len(all_files)}")
        print(f"Actual DICOM files: {len(dicom_files)}")

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
                            ""
                        ),
                        "rows": ds.get(
                            "Rows",
                            ""
                        ),
                        "columns": ds.get(
                            "Columns",
                            ""
                        ),
                    }
                )

            except Exception as e:
                print(
                    f"ERROR reading {filename}: {e}"
                )

            if index % 100 == 0:
                print(
                    f"Processed {index}/{len(dicom_files)}"
                )

    xray = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("Mapping summary")
    print("=" * 70)

    print(f"DICOM records: {len(xray)}")
    print(f"Unique patients: {xray['patient_id'].nunique()}")

    print("\nView distribution:")
    print(
        xray["view"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nUnknown views:")
    print(
        (xray["view"] == "Unknown").sum()
    )

    print("\nPatients by view:")
    patient_views = (
        xray
        .groupby(["patient_id", "view"])
        .size()
        .unstack(fill_value=0)
    )

    for column in [
        "AP",
        "Lateral",
        "Extension",
        "Unknown"
    ]:
        if column not in patient_views.columns:
            patient_views[column] = 0

    print(
        f"Patients with AP: "
        f"{(patient_views['AP'] > 0).sum()}"
    )

    print(
        f"Patients with Lateral: "
        f"{(patient_views['Lateral'] > 0).sum()}"
    )

    print(
        f"Patients with AP + Lateral: "
        f"{((patient_views['AP'] > 0) & (patient_views['Lateral'] > 0)).sum()}"
    )

    xray = xray.sort_values(
        [
            "patient_id",
            "view",
            "series_instance_uid",
            "instance_number",
            "dicom_file",
        ]
    )

    xray.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
