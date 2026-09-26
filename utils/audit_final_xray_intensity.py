import zipfile
import numpy as np
import pandas as pd
import pydicom


MANIFEST = "results/final_multimodal_manifest.csv"
MAPPING = "results/xray_image_mapping.csv"
ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"


manifest = pd.read_csv(MANIFEST)
mapping = pd.read_csv(MAPPING)

records = []

for _, row in manifest.iterrows():
    records.append({
        "patient_id": int(row["patient_id"]),
        "view": "AP",
        "dicom_file": row["ap_dicom_file"],
    })
    records.append({
        "patient_id": int(row["patient_id"]),
        "view": "Lateral",
        "dicom_file": row["lateral_dicom_file"],
    })

records = pd.DataFrame(records)

records = records.merge(
    mapping[
        [
            "patient_id",
            "dicom_file",
            "rows",
            "columns",
        ]
    ],
    on=["patient_id", "dicom_file"],
    how="left",
)

if records[["rows", "columns"]].isna().any().any():
    raise RuntimeError("Some X-rays could not be matched to the mapping.")


results = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for _, row in records.iterrows():

        with z.open(row["dicom_file"]) as f:
            ds = pydicom.dcmread(f)

        image = ds.pixel_array.astype(np.float32)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            bits_stored = int(ds.BitsStored)
            max_value = float((1 << bits_stored) - 1)
            image = max_value - image

        p1, p50, p99 = np.percentile(
            image,
            [1, 50, 99],
        )

        results.append({
            "patient_id": int(row["patient_id"]),
            "view": row["view"],
            "bits_stored": int(ds.BitsStored),
            "photometric": str(
                getattr(ds, "PhotometricInterpretation", "")
            ),
            "minimum": float(image.min()),
            "maximum": float(image.max()),
            "mean": float(image.mean()),
            "p1": float(p1),
            "median": float(p50),
            "p99": float(p99),
        })


audit = pd.DataFrame(results)

print("Final multimodal X-ray intensity audit")
print("======================================")
print(f"Images: {len(audit)}")
print()

print("BitsStored:")
print(audit["bits_stored"].value_counts().sort_index().to_dict())
print()

print("Photometric interpretation:")
print(audit["photometric"].value_counts().to_dict())
print()

print("Raw intensity summary:")
print(
    audit[
        [
            "minimum",
            "maximum",
            "mean",
            "p1",
            "median",
            "p99",
        ]
    ].describe().round(2).to_string()
)

print()

print("P1 range:")
print(
    f"{audit['p1'].min():.1f} "
    f"to "
    f"{audit['p1'].max():.1f}"
)

print("P99 range:")
print(
    f"{audit['p99'].min():.1f} "
    f"to "
    f"{audit['p99'].max():.1f}"
)

print()

print("Highest P99 images:")
print(
    audit.nlargest(5, "p99")[
        [
            "patient_id",
            "view",
            "bits_stored",
            "minimum",
            "maximum",
            "p1",
            "p99",
        ]
    ].to_string(index=False)
)

print()

print("Lowest P99 images:")
print(
    audit.nsmallest(5, "p99")[
        [
            "patient_id",
            "view",
            "bits_stored",
            "minimum",
            "maximum",
            "p1",
            "p99",
        ]
    ].to_string(index=False)
)
