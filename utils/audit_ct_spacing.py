import zipfile
import pandas as pd
import pydicom
import numpy as np


def get_zip_path(name):
    return "dataset/" + str(name)


manifest = pd.read_csv("results/final_multimodal_manifest.csv")

all_spacings = []
patient_results = []

for _, row in manifest.iterrows():
    zip_path = get_zip_path(row["ct_zip"])
    patient_folder = row["ct_patient_folder"]

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and f"{patient_folder}/" in n
        ]

        z_positions = []

        for name in names:
            with z.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            if hasattr(ds, "ImagePositionPatient"):
                z_positions.append(float(ds.ImagePositionPatient[2]))

        z_positions = sorted(z_positions)

        if len(z_positions) >= 2:
            spacings = np.diff(z_positions)
            spacing = float(np.median(np.abs(spacings)))

            all_spacings.extend(np.abs(spacings))

            patient_results.append(
                {
                    "patient_id": int(row["patient_id"]),
                    "slices": len(z_positions),
                    "median_spacing_mm": spacing,
                    "min_spacing_mm": float(np.min(np.abs(spacings))),
                    "max_spacing_mm": float(np.max(np.abs(spacings))),
                }
            )

print("CT spacing audit")
print("================")
print("Patients checked:", len(patient_results))

patient_df = pd.DataFrame(patient_results)

print("\nPer-patient median spacing:")
print(
    patient_df["median_spacing_mm"]
    .describe()
    .round(3)
)

print("\nOverall spacing distribution:")
print(
    pd.Series(all_spacings)
    .describe()
    .round(3)
)

print("\nUnique rounded median spacings:")
print(
    patient_df["median_spacing_mm"]
    .round(3)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nPatients with variable spacing:")
variable = patient_df[
    (patient_df["min_spacing_mm"].round(3)
     != patient_df["max_spacing_mm"].round(3))
]

print("Count:", len(variable))

if len(variable) > 0:
    print(variable.head(20).to_string(index=False))
