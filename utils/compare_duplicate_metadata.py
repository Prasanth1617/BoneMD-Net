import zipfile
import pandas as pd
import pydicom

manifest = pd.read_csv("results/final_multimodal_manifest.csv")

patients = [10, 146, 168]

print("CT Duplicate Metadata Comparison")
print("================================")

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
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            image_type = str(getattr(ds, "ImageType", ""))

            if (
                "ORIGINAL" in image_type
                and "PRIMARY" in image_type
                and "AXIAL" in image_type
            ):
                records.append(
                    {
                        "file": name,
                        "z": float(ds.ImagePositionPatient[2]),
                        "instance": getattr(ds, "InstanceNumber", None),
                        "series_description": str(
                            getattr(ds, "SeriesDescription", "")
                        ),
                        "slice_thickness": getattr(
                            ds, "SliceThickness", None
                        ),
                        "pixel_spacing": str(
                            getattr(ds, "PixelSpacing", None)
                        ),
                        "image_orientation": str(
                            getattr(ds, "ImageOrientationPatient", None)
                        ),
                        "kernel": str(
                            getattr(ds, "ConvolutionKernel", "")
                        ),
                        "kvp": getattr(ds, "KVP", None),
                        "exposure": getattr(ds, "Exposure", None),
                        "exposure_time": getattr(
                            ds, "ExposureTime", None
                        ),
                        "tube_current": getattr(
                            ds, "XRayTubeCurrent", None
                        ),
                        "manufacturer": str(
                            getattr(ds, "Manufacturer", "")
                        ),
                        "model": str(
                            getattr(ds, "ManufacturerModelName", "")
                        ),
                        "protocol": str(
                            getattr(ds, "ProtocolName", "")
                        ),
                        "study_description": str(
                            getattr(ds, "StudyDescription", "")
                        ),
                    }
                )

    df = pd.DataFrame(records)

    duplicate_z = (
        df.groupby("z")
        .filter(lambda g: len(g) > 1)
        .sort_values(["z", "instance"])
    )

    print(f"\nPatient {patient_id}")
    print("-" * 70)

    if duplicate_z.empty:
        print("No duplicate Z positions found.")
        continue

    z_value = duplicate_z.iloc[0]["z"]
    pair = duplicate_z[duplicate_z["z"] == z_value]

    print("Example duplicate Z position:", z_value)
    print("\nMetadata for the two slices:\n")

    columns = [
        "file",
        "instance",
        "series_description",
        "slice_thickness",
        "pixel_spacing",
        "image_orientation",
        "kernel",
        "kvp",
        "exposure",
        "exposure_time",
        "tube_current",
        "manufacturer",
        "model",
        "protocol",
        "study_description",
    ]

    print(pair[columns].to_string(index=False))

print("\nComparison complete.")
