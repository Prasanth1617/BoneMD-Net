import zipfile
from pathlib import Path

import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"


def main():

    manifest = pd.read_csv(MANIFEST)

    problems = []

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

                spacing = getattr(
                    ds,
                    "PixelSpacing",
                    None,
                )

                if position is None or spacing is None:
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

        df = pd.DataFrame(records)

        if df.empty:
            continue

        candidates = []

        for (
            kernel,
            thickness,
            row_spacing,
            col_spacing,
        ), group in df.groupby(
            [
                "kernel",
                "thickness",
                "row_spacing",
                "col_spacing",
            ]
        ):

            z_values = sorted(group["z"].unique())

            candidates.append(
                {
                    "kernel": kernel,
                    "thickness": thickness,
                    "row_spacing": row_spacing,
                    "col_spacing": col_spacing,
                    "slice_count": len(group),
                    "unique_z": len(z_values),
                    "z_min": min(z_values),
                    "z_max": max(z_values),
                }
            )

        candidates_df = pd.DataFrame(candidates)

        candidates_df["coverage"] = (
            candidates_df["z_max"]
            - candidates_df["z_min"]
        )

        candidates_df = candidates_df.sort_values(
            by=[
                "thickness",
                "row_spacing",
                "col_spacing",
                "coverage",
                "kernel",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
            ],
        )

        selected = candidates_df.iloc[0]

        if selected["slice_count"] != selected["unique_z"]:

            problems.append(
                {
                    "patient_id": patient_id,
                    "kernel": selected["kernel"],
                    "thickness": selected["thickness"],
                    "slice_count": int(
                        selected["slice_count"]
                    ),
                    "unique_z": int(
                        selected["unique_z"]
                    ),
                    "duplicate_slices": int(
                        selected["slice_count"]
                        - selected["unique_z"]
                    ),
                }
            )

    print()
    print("=" * 70)
    print("SELECTED CT DUPLICATE-Z AUDIT")
    print("=" * 70)

    print(
        f"Patients with duplicate Z positions "
        f"in selected reconstruction: {len(problems)}"
    )

    print()

    if problems:
        print(
            pd.DataFrame(problems)
            .to_string(index=False)
        )
    else:
        print(
            "No duplicate Z positions found "
            "in selected reconstructions."
        )


if __name__ == "__main__":
    main()
