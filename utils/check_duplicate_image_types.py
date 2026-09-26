import zipfile
import pandas as pd
import pydicom

manifest = pd.read_csv("results/final_multimodal_manifest.csv")

patients = [10, 146, 168]

print("Duplicate Patient ImageType Check")
print("=================================")

for patient_id in patients:
    row = manifest[manifest["patient_id"] == patient_id].iloc[0]

    zip_path = "dataset/" + str(row["ct_zip"])
    patient_folder = row["ct_patient_folder"]

    image_types = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [
            n
            for n in z.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and f"{patient_folder}/" in n
        ]

        for name in names:
            with z.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            image_types.append(str(getattr(ds, "ImageType", "")))

    print(f"\nPatient {patient_id}")
    print("-" * 60)
    print("Total DICOMs:", len(image_types))
    print("Unique ImageType values:")

    counts = pd.Series(image_types).value_counts()

    for image_type, count in counts.items():
        print(f"{count:4d}  {image_type}")

print("\nCheck complete.")
