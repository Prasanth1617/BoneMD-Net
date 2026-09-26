import pandas as pd
import numpy as np


MANIFEST = "results/final_multimodal_manifest.csv"
TARGET_SIZE = 224

df = pd.read_csv(MANIFEST)

records = []

for _, row in df.iterrows():
    for view, column in [
        ("AP", "ap_dicom_file"),
        ("Lateral", "lateral_dicom_file"),
    ]:
        dicom_file = row[column]

        # Get dimensions from the mapping later
        records.append({
            "patient_id": int(row["patient_id"]),
            "view": view,
            "dicom_file": dicom_file,
        })

mapping = pd.read_csv("results/xray_image_mapping.csv")

records_df = pd.DataFrame(records)

records_df = records_df.merge(
    mapping[
        ["patient_id", "dicom_file", "rows", "columns"]
    ],
    on=["patient_id", "dicom_file"],
    how="left",
)

if records_df[["rows", "columns"]].isna().any().any():
    raise RuntimeError("Some final-manifest X-rays could not be matched.")

scale = np.minimum(
    TARGET_SIZE / records_df["rows"],
    TARGET_SIZE / records_df["columns"],
)

records_df["new_rows"] = np.round(
    records_df["rows"] * scale
).astype(int)

records_df["new_cols"] = np.round(
    records_df["columns"] * scale
).astype(int)

records_df["padding_fraction"] = 1 - (
    records_df["new_rows"] * records_df["new_cols"]
) / (TARGET_SIZE * TARGET_SIZE)

print("Final multimodal X-ray resize/padding audit")
print("============================================")
print(f"Patients: {df['patient_id'].nunique()}")
print(f"X-ray images: {len(records_df)}")
print(f"Target size: {TARGET_SIZE} x {TARGET_SIZE}")
print()

print("Padding fraction:")
print(f"Minimum: {records_df['padding_fraction'].min():.3f}")
print(f"Median:  {records_df['padding_fraction'].median():.3f}")
print(f"Mean:    {records_df['padding_fraction'].mean():.3f}")
print(f"Maximum: {records_df['padding_fraction'].max():.3f}")
print()

print(
    "Images with >50% padding:",
    int((records_df["padding_fraction"] > 0.50).sum())
)

print(
    "Images with >60% padding:",
    int((records_df["padding_fraction"] > 0.60).sum())
)

print()

print("Padding by view:")
print(
    records_df.groupby("view")["padding_fraction"]
    .agg(["count", "mean", "median", "max"])
    .round(3)
)

print()

print("Worst 10 final-manifest images:")
print(
    records_df.nlargest(10, "padding_fraction")[
        [
            "patient_id",
            "view",
            "rows",
            "columns",
            "new_rows",
            "new_cols",
            "padding_fraction",
        ]
    ].to_string(index=False)
)
