import zipfile
import pandas as pd
import pydicom

manifest = pd.read_csv("results/final_multimodal_manifest.csv")

affected_patients = [10, 63, 76, 138, 146, 168, 232, 262]

print("CT Series Metadata Audit")
print("========================")

for patient_id in affected_patients:
    row = manifest[manifest["patient_id"] == patient_id].iloc[0]

    zip_path = "dataset/" + str(row["ct_zip"])
    patient_folder = row["ct_patient_folder"]

    records = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and f"{patient_folder}/" in n
        ]

        for name in names:
            with z.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            records.append({
                "file": name,
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "series_description": str(
                    getattr(ds, "SeriesDescription", "")
                ),
                "instance_number": getattr(ds, "InstanceNumber", None),
                "image_type": str(getattr(ds, "ImageType", "")),
                "slice_thickness": getattr(ds, "SliceThickness", None),
                "rows": getattr(ds, "Rows", None),
                "columns": getattr(ds, "Columns", None),
            })

    df = pd.DataFrame(records)

    print(f"\nPatient {patient_id}")
    print("-" * 60)
    print("Total DICOMs:", len(df))
    print("Unique SeriesInstanceUIDs:", df["series_uid"].nunique())
    print("Unique StudyInstanceUIDs:", df["study_uid"].nunique())

    summary = (
        df.groupby(
            [
                "series_uid",
                "study_uid",
                "series_description",
                "image_type",
                "slice_thickness",
                "rows",
                "columns",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="slice_count")
    )

    print("\nSeries summary:")
    print(summary.to_string(index=False))
