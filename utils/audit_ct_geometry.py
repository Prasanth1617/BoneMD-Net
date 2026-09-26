import zipfile
import pandas as pd
import pydicom

manifest = pd.read_csv("results/final_multimodal_manifest.csv")

affected_patients = [10, 63, 76, 138, 146, 168, 232, 262]

print("CT Geometry Audit")
print("=================")

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

            position = getattr(ds, "ImagePositionPatient", None)
            orientation = getattr(ds, "ImageOrientationPatient", None)

            records.append({
                "file": name,
                "instance": getattr(ds, "InstanceNumber", None),
                "z": float(position[2]) if position is not None else None,
                "x": float(position[0]) if position is not None else None,
                "y": float(position[1]) if position is not None else None,
                "orientation": str(orientation),
                "image_type": str(getattr(ds, "ImageType", "")),
                "slice_thickness": getattr(ds, "SliceThickness", None),
                "rows": getattr(ds, "Rows", None),
                "columns": getattr(ds, "Columns", None),
                "series_description": str(
                    getattr(ds, "SeriesDescription", "")
                ),
            })

    df = pd.DataFrame(records)

    print(f"\nPatient {patient_id}")
    print("-" * 70)

    # Primary axial candidates
    axial = df[
        df["image_type"].str.contains(
            "ORIGINAL.*PRIMARY.*AXIAL",
            regex=True,
            na=False
        )
        & (df["rows"] == 512)
        & (df["columns"] == 512)
    ].copy()

    axial = axial.sort_values("z")

    print("Total DICOMs:", len(df))
    print("Primary axial candidates:", len(axial))

    if len(axial) > 0:
        print("Z minimum:", axial["z"].min())
        print("Z maximum:", axial["z"].max())

        z = axial["z"].dropna().sort_values().to_numpy()

        if len(z) > 1:
            diffs = z[1:] - z[:-1]

            print("Median Z spacing:", round(float(pd.Series(diffs).median()), 4))
            print("Minimum Z spacing:", round(float(diffs.min()), 4))
            print("Maximum Z spacing:", round(float(diffs.max()), 4))

            print("Unique rounded spacings:")
            print(
                pd.Series(diffs)
                .round(4)
                .value_counts()
                .sort_index()
                .to_string()
            )

        print("\nFirst 5 axial slices:")
        print(
            axial[
                ["file", "instance", "z", "slice_thickness"]
            ].head(5).to_string(index=False)
        )

        print("\nLast 5 axial slices:")
        print(
            axial[
                ["file", "instance", "z", "slice_thickness"]
            ].tail(5).to_string(index=False)
        )

    else:
        print("WARNING: No primary axial candidates found.")
