import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zipfile
import numpy as np
import pandas as pd
import pydicom

from utils.ct_loader import load_ct_volume


CT_ZIPS = {
    1: "dataset/lumos_ct_001_070_dcm.zip",
    71: "dataset/lumos_ct_071_140_dcm.zip",
    141: "dataset/lumos_ct_141_210_dcm.zip",
    211: "dataset/lumos_ct_211_280_dcm.zip",
}


def find_zip(patient_id):
    for start_id, path in CT_ZIPS.items():
        if start_id <= patient_id < start_id + 70:
            return path
    raise ValueError(f"No CT ZIP for patient {patient_id}")


def main():

    manifest_path = "results/final_multimodal_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    patient_ids = sorted(manifest["patient_id"].astype(int).unique())

    results = []

    print()
    print("=" * 110)
    print("ALL-PATIENT CT PADDING AUDIT")
    print("=" * 110)
    print(f"Patients to inspect: {len(patient_ids)}")

    for i, patient_id in enumerate(patient_ids, start=1):

        patient_folder = f"lumos_ct_{patient_id:03d}"
        zip_path = find_zip(patient_id)

        try:

            with zipfile.ZipFile(zip_path, "r") as archive:

                names = [
                    name
                    for name in archive.namelist()
                    if name.startswith(patient_folder + "/")
                    and name.lower().endswith((".dcm", ".dicom"))
                ]

                if not names:
                    raise RuntimeError("No DICOM files found")

                raw_padding_values = []

                for name in names:

                    with archive.open(name) as f:
                        ds = pydicom.dcmread(
                            f,
                            stop_before_pixels=False
                        )

                    raw = ds.pixel_array

                    # Candidate padding values are the extreme raw values.
                    raw_min = int(raw.min())

                    count_min = int(np.count_nonzero(raw == raw_min))
                    fraction_min = count_min / raw.size

                    slope = float(
                        getattr(ds, "RescaleSlope", 1.0)
                    )
                    intercept = float(
                        getattr(ds, "RescaleIntercept", 0.0)
                    )

                    hu_min = raw_min * slope + intercept

                    raw_padding_values.append(
                        (
                            raw_min,
                            hu_min,
                            fraction_min,
                        )
                    )

                raw_values = [x[0] for x in raw_padding_values]

                value_counts = pd.Series(raw_values).value_counts()

                most_common_raw = int(value_counts.index[0])
                most_common_count = int(value_counts.iloc[0])
                most_common_fraction_slices = (
                    most_common_count / len(raw_values)
                )

                # Use the first DICOM to obtain representative rescale metadata.
                with archive.open(names[0]) as f:
                    first_ds = pydicom.dcmread(f)

                slope = float(
                    getattr(first_ds, "RescaleSlope", 1.0)
                )
                intercept = float(
                    getattr(first_ds, "RescaleIntercept", 0.0)
                )

                padding_hu = (
                    most_common_raw * slope + intercept
                )

                # Count pixels equal to the most common extreme value
                # across the entire reconstruction.
                total_pixels = 0
                padding_pixels = 0

                for name in names:

                    with archive.open(name) as f:
                        ds = pydicom.dcmread(f)

                    raw = ds.pixel_array

                    total_pixels += raw.size
                    padding_pixels += int(
                        np.count_nonzero(raw == most_common_raw)
                    )

                padding_fraction = (
                    padding_pixels / total_pixels
                )

                results.append(
                    {
                        "patient_id": patient_id,
                        "slice_count": len(names),
                        "raw_padding_value": most_common_raw,
                        "padding_hu": padding_hu,
                        "padding_fraction": padding_fraction,
                        "slices_with_padding_value_fraction":
                            most_common_fraction_slices,
                        "rescale_slope": slope,
                        "rescale_intercept": intercept,
                    }
                )

                print(
                    f"[{i:3d}/{len(patient_ids)}] "
                    f"patient={patient_id:3d} "
                    f"slices={len(names):3d} "
                    f"raw={most_common_raw:6d} "
                    f"HU={padding_hu:8.1f} "
                    f"padding={padding_fraction * 100:6.2f}%"
                )

        except Exception as e:

            print(
                f"[{i:3d}/{len(patient_ids)}] "
                f"patient={patient_id:3d} "
                f"ERROR: {e}"
            )

    df = pd.DataFrame(results)

    output_path = "results/ct_padding_audit.csv"
    df.to_csv(output_path, index=False)

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print("Patients successfully audited:", len(df))

    if len(df) > 0:

        print()
        print("Raw padding values:")
        print(
            df["raw_padding_value"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print("Padding HU values:")
        print(
            df["padding_hu"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print("Padding fraction (%):")
        print(
            df["padding_fraction"]
            .mul(100)
            .describe()
            .to_string()
        )

        print()
        print("Patients with >10% padding:")
        print(
            int(
                (df["padding_fraction"] > 0.10).sum()
            )
        )

        print()
        print("Patients with >20% padding:")
        print(
            int(
                (df["padding_fraction"] > 0.20).sum()
            )
        )

        print()
        print("Patients with padding HU < -1500:")
        print(
            int(
                (df["padding_hu"] < -1500).sum()
            )
        )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
