import zipfile
from pathlib import Path

import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"

TARGET_PATIENTS = [10, 15, 76, 168, 268]


def main():

    manifest = pd.read_csv(MANIFEST)

    for _, row in manifest[
        manifest["patient_id"].isin(TARGET_PATIENTS)
    ].iterrows():

        patient_id = int(row["patient_id"])
        zip_path = PROJECT_ROOT / "dataset" / row["ct_zip"]
        patient_folder = row["ct_patient_folder"]

        print()
        print("=" * 70)
        print(f"PATIENT {patient_id:03d}")
        print("=" * 70)

        records = []

        with zipfile.ZipFile(zip_path, "r") as z:

            names = [
                n
                for n in z.namelist()
                if n.lower().endswith(".dcm")
                and "__macosx" not in n.lower()
                and n.startswith(patient_folder + "/")
            ]

            for name in names:

                with z.open(name) as f:
                    ds = pydicom.dcmread(
                        f,
                        stop_before_pixels=True,
                        specific_tags=[
                            "ImagePositionPatient",
                            "ConvolutionKernel",
                            "ImageType",
                            "SeriesDescription",
                            "SliceThickness",
                            "PixelSpacing",
                        ],
                    )

                image_type = [
                    str(x)
                    for x in getattr(ds, "ImageType", [])
                ]

                position = getattr(
                    ds,
                    "ImagePositionPatient",
                    None,
                )

                if position is None:
                    continue

                records.append(
                    {
                        "z": float(position[2]),
                        "kernel": str(
                            getattr(
                                ds,
                                "ConvolutionKernel",
                                "Unknown",
                            )
                        ),
                        "image_type": "\\".join(image_type),
                        "series": str(
                            getattr(
                                ds,
                                "SeriesDescription",
                                "Unknown",
                            )
                        ),
                        "thickness": float(
                            getattr(
                                ds,
                                "SliceThickness",
                                0,
                            )
                        ),
                        "pixel_spacing": str(
                            getattr(
                                ds,
                                "PixelSpacing",
                                "Unknown",
                            )
                        ),
                    }
                )

        df = pd.DataFrame(records)

        primary = df[
            df["image_type"].str.contains(
                "PRIMARY",
                na=False,
            )
            & df["image_type"].str.contains(
                "AXIAL",
                na=False,
            )
        ].copy()

        print(f"Total DICOMs: {len(df)}")
        print(f"Primary axial DICOMs: {len(primary)}")
        print()

        print("Reconstruction groups:")
        print("-" * 70)

        summary = (
            primary
            .groupby(
                [
                    "kernel",
                    "series",
                    "thickness",
                    "pixel_spacing",
                ]
            )
            .agg(
                count=("z", "size"),
                z_min=("z", "min"),
                z_max=("z", "max"),
            )
            .reset_index()
        )

        print(summary.to_string(index=False))

        print()
        print("Kernel-specific spatial coverage:")
        print("-" * 70)

        for kernel, group in primary.groupby("kernel"):

            z_values = sorted(
                group["z"].dropna().unique()
            )

            if len(z_values) == 1:
                spacing_text = "N/A"
            else:
                spacings = [
                    round(
                        z_values[i + 1] - z_values[i],
                        3,
                    )
                    for i in range(len(z_values) - 1)
                ]

                spacing_text = (
                    f"{min(spacings):.3f} "
                    f"to {max(spacings):.3f} mm"
                )

            print(
                f"{kernel}: "
                f"{len(group)} slices, "
                f"{len(z_values)} unique Z positions, "
                f"Z {min(z_values):.3f} "
                f"to {max(z_values):.3f}, "
                f"spacing {spacing_text}"
            )


if __name__ == "__main__":
    main()