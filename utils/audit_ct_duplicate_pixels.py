import zipfile
import numpy as np
import pandas as pd
import pydicom

manifest = pd.read_csv("results/final_multimodal_manifest.csv")

patients = [10, 146, 168]

print("CT Duplicate Pixel Audit")
print("========================")

for patient_id in patients:
    row = manifest[manifest["patient_id"] == patient_id].iloc[0]

    zip_path = "dataset/" + str(row["ct_zip"])
    patient_folder = row["ct_patient_folder"]

    records = []

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
                ds = pydicom.dcmread(f)

            image_type = str(getattr(ds, "ImageType", ""))

            if (
    "ORIGINAL" in image_type
    and "PRIMARY" in image_type
    and "AXIAL" in image_type
):
                z_pos = float(ds.ImagePositionPatient[2])

                records.append(
                    {
                        "file": name,
                        "z": z_pos,
                        "instance": getattr(ds, "InstanceNumber", None),
                        "pixels": ds.pixel_array.astype(np.float32),
                    }
                )

    print(f"\nPatient {patient_id}")
    print("-" * 60)
    print("Primary axial slices:", len(records))

    if not records:
        print("WARNING: No primary axial slices found.")
        continue

    df = pd.DataFrame(records)

    duplicate_groups = (
        df.groupby("z")
        .filter(lambda g: len(g) > 1)
        .groupby("z")
    )

    print("Duplicate Z positions:", duplicate_groups.ngroups)

    identical = 0
    different = 0
    max_difference = 0.0

    for z_pos, group in duplicate_groups:
        images = list(group["pixels"])

        if len(images) != 2:
            print(
                f"WARNING: Z={z_pos} has {len(images)} copies"
            )
            continue

        diff = np.max(np.abs(images[0] - images[1]))

        if diff == 0:
            identical += 1
        else:
            different += 1

        max_difference = max(max_difference, float(diff))

    print("Identical duplicate pairs:", identical)
    print("Different duplicate pairs:", different)
    print("Maximum pixel difference:", max_difference)

print("\nAudit complete.")
