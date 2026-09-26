import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ct_loader import load_ct_volume


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"
OUTPUT = PROJECT_ROOT / "results" / "selected_ct_padding_audit.csv"


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = []

    for _, row in manifest.iterrows():
        patient_id = int(row["patient_id"])

        zip_path = PROJECT_ROOT / "dataset" / str(row["ct_zip"])
        patient_folder = str(row["ct_patient_folder"])

        try:
            volume, metadata = load_ct_volume(
                zip_path,
                patient_folder,
            )

            total_pixels = volume.size

            fractions = {
                "lt_-1500": float(np.mean(volume < -1500)),
                "lt_-1200": float(np.mean(volume < -1200)),
                "lt_-1000": float(np.mean(volume < -1000)),
                "eq_-2048": float(np.mean(volume == -2048)),
                "eq_-3024": float(np.mean(volume == -3024)),
            }

            rows.append(
                {
                    "patient_id": patient_id,
                    "shape": str(volume.shape),
                    "kernel": metadata[0].get("kernel") if metadata else None,
                    "slice_thickness": metadata[0].get("thickness") if metadata else None,
                    "slice_count": volume.shape[0],
                    "hu_min": float(volume.min()),
                    "hu_max": float(volume.max()),
                    "hu_mean": float(volume.mean()),
                    "fraction_lt_-1500": fractions["lt_-1500"],
                    "fraction_lt_-1200": fractions["lt_-1200"],
                    "fraction_lt_-1000": fractions["lt_-1000"],
                    "fraction_eq_-2048": fractions["eq_-2048"],
                    "fraction_eq_-3024": fractions["eq_-3024"],
                }
            )

            print(
                f"{patient_id:03d} "
                f"shape={volume.shape} "
                f"kernel={metadata[0].get('kernel') if metadata else None} "
                f"thickness={metadata[0].get('thickness') if metadata else None} "
                f"min={volume.min():.1f} "
                f"<-1500={fractions['lt_-1500'] * 100:.2f}%"
            )

        except Exception as e:
            print(f"{patient_id:03d} ERROR: {e}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    print()
    print("=" * 70)
    print("SELECTED CT PADDING AUDIT")
    print("=" * 70)
    print(f"Patients audited: {len(result)}")

    if len(result):
        print(
            f"Patients with HU < -1500: "
            f"{(result['fraction_lt_-1500'] > 0).sum()}"
        )

        print(
            f"Patients with >10% HU < -1500: "
            f"{(result['fraction_lt_-1500'] > 0.10).sum()}"
        )

        print(
            f"Patients with >20% HU < -1500: "
            f"{(result['fraction_lt_-1500'] > 0.20).sum()}"
        )

        print()
        print("Highest extreme-negative fractions:")

        top = result.sort_values(
            "fraction_lt_-1500",
            ascending=False
        ).head(20)

        print(
            top[
                [
                    "patient_id",
                    "kernel",
                    "slice_thickness",
                    "fraction_lt_-1500",
                    "fraction_eq_-2048",
                    "fraction_eq_-3024",
                ]
            ].to_string(index=False)
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
