import zipfile
from pathlib import Path

import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"


def main():

    manifest = pd.read_csv(MANIFEST)

    print("CT Reconstruction Geometry Audit")
    print("=" * 70)
    print(f"Eligible patients: {len(manifest)}")
    print()

    multi_group_patients = []

    for _, row in manifest.iterrows():

        patient_id = int(row["patient_id"])
        zip_path = PROJECT_ROOT / "dataset" / row["ct_zip"]
        patient_folder = row["ct_patient_folder"]

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
                            "SliceThickness",
                            "PixelSpacing",
                        ],
                    )

                image_type = [
                    str(x)
                    for x in getattr(ds, "ImageType", [])
                ]

                if "PRIMARY" not in image_type:
                    continue

                if "AXIAL" not in image_type:
                    continue

                position = getattr(
                    ds,
                    "ImagePositionPatient",
                    None,
                )

                if position is None:
                    continue

                spacing = getattr(
                    ds,
                    "PixelSpacing",
                    None,
                )

                if spacing is None:
                    continue

                records.append(
                    {
                        "z": round(float(position[2]), 3),
                        "kernel": str(
                            getattr(
                                ds,
                                "ConvolutionKernel",
                                "Unknown",
                            )
                        ),
                        "thickness": round(
                            float(
                                getattr(
                                    ds,
                                    "SliceThickness",
                                    0,
                                )
                            ),
                            3,
                        ),
                        "row_spacing": round(
                            float(spacing[0]),
                            6,
                        ),
                        "col_spacing": round(
                            float(spacing[1]),
                            6,
                        ),
                    }
                )

        if not records:
            continue

        df = pd.DataFrame(records)

        groups = []

        for (
            thickness,
            row_spacing,
            col_spacing,
        ), group in df.groupby(
            [
                "thickness",
                "row_spacing",
                "col_spacing",
            ]
        ):

            z_values = sorted(
                group["z"].unique()
            )

            if len(z_values) > 1:

                spacings = [
                    round(
                        z_values[i + 1] - z_values[i],
                        3,
                    )
                    for i in range(
                        len(z_values) - 1
                    )
                ]

                median_spacing = round(
                    float(
                        pd.Series(spacings).median()
                    ),
                    3,
                )

            else:
                median_spacing = 0.0

            kernels = sorted(
                group["kernel"].unique()
            )

            groups.append(
                {
                    "thickness": thickness,
                    "row_spacing": row_spacing,
                    "col_spacing": col_spacing,
                    "unique_z": len(z_values),
                    "z_min": min(z_values),
                    "z_max": max(z_values),
                    "median_z_spacing": median_spacing,
                    "kernels": ", ".join(kernels),
                }
            )

        if len(groups) > 1:
            multi_group_patients.append(
                (patient_id, groups)
            )

    print(
        f"Patients with multiple geometry groups: "
        f"{len(multi_group_patients)}"
    )
    print()

    if not multi_group_patients:
        print(
            "No patients have multiple primary-axial "
            "geometry groups."
        )
        return

    for patient_id, groups in multi_group_patients:

        print("=" * 70)
        print(f"PATIENT {patient_id:03d}")
        print("=" * 70)

        result = pd.DataFrame(groups)

        print(
            result.to_string(
                index=False
            )
        )

        print()


if __name__ == "__main__":
    main()
