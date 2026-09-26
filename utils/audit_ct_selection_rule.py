import zipfile
from pathlib import Path

import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"


def main():

    manifest = pd.read_csv(MANIFEST)

    selections = []

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
                        "file": name,
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
            print(f"ERROR: Patient {patient_id:03d} has no primary axial CT")
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

        # Selection priority:
        # 1. Smaller slice thickness
        # 2. Smaller in-plane pixel spacing
        # 3. Larger spatial coverage
        # 4. Deterministic kernel name
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

        selections.append(
            {
                "patient_id": patient_id,
                "selected_kernel": selected["kernel"],
                "selected_thickness": selected["thickness"],
                "selected_row_spacing": selected["row_spacing"],
                "selected_col_spacing": selected["col_spacing"],
                "selected_slices": selected["slice_count"],
                "selected_unique_z": selected["unique_z"],
                "selected_z_min": selected["z_min"],
                "selected_z_max": selected["z_max"],
                "candidate_count": len(candidates_df),
            }
        )

    result = pd.DataFrame(selections)

    print()
    print("=" * 70)
    print("CT SELECTION RULE AUDIT")
    print("=" * 70)

    print()
    print(f"Patients processed: {len(result)}")
    print(
        "Patients with multiple reconstruction candidates:",
        int((result["candidate_count"] > 1).sum()),
    )

    print()
    print("Selected slice-thickness distribution:")
    print(
        result["selected_thickness"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Patients with multiple candidates:")
    print("-" * 70)

    multi = result[result["candidate_count"] > 1]

    if multi.empty:
        print("None")
    else:
        print(
            multi[
                [
                    "patient_id",
                    "selected_kernel",
                    "selected_thickness",
                    "selected_slices",
                    "selected_z_min",
                    "selected_z_max",
                    "candidate_count",
                ]
            ].to_string(index=False)
        )

    print()
    print("Selected reconstruction summary:")
    print("-" * 70)

    print(
        result[
            [
                "patient_id",
                "selected_kernel",
                "selected_thickness",
                "selected_row_spacing",
                "selected_slices",
                "selected_z_min",
                "selected_z_max",
            ]
        ].head(20).to_string(index=False)
    )

    output_path = (
        PROJECT_ROOT
        / "results"
        / "ct_selection_audit.csv"
    )

    result.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
