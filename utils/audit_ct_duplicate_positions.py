import zipfile
import pandas as pd
import pydicom
import numpy as np


manifest = pd.read_csv("results/final_multimodal_manifest.csv")

print("CT duplicate/near-duplicate Z-position audit")
print("=============================================")

found = []

for _, row in manifest.iterrows():
    zip_path = "dataset/" + str(row["ct_zip"])
    patient_folder = row["ct_patient_folder"]

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and f"{patient_folder}/" in n
        ]

        positions = []

        for name in names:
            with z.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            if hasattr(ds, "ImagePositionPatient"):
                z_pos = float(ds.ImagePositionPatient[2])
                positions.append((z_pos, name))

        positions.sort()

        for i in range(1, len(positions)):
            diff = abs(positions[i][0] - positions[i - 1][0])

            if diff < 0.1:
                found.append(
                    {
                        "patient_id": int(row["patient_id"]),
                        "difference_mm": diff,
                        "slice_1": positions[i - 1][1],
                        "z_1": positions[i - 1][0],
                        "slice_2": positions[i][1],
                        "z_2": positions[i][0],
                    }
                )

print("Near-duplicate pairs (< 0.1 mm):", len(found))

if found:
    df = pd.DataFrame(found)

    print("\nPatients affected:", df["patient_id"].nunique())
    print("\nDetails:")
    print(df.to_string(index=False))

    print("\nSaved report:")
    df.to_csv("results/ct_duplicate_positions.csv", index=False)
    print("results/ct_duplicate_positions.csv")
else:
    print("No near-duplicate positions found.")
