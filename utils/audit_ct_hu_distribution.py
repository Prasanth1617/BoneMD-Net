import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from utils.ct_loader import load_ct_volume


TESTS = [
    ("dataset/lumos_ct_001_070_dcm.zip", "lumos_ct_001"),
    ("dataset/lumos_ct_001_070_dcm.zip", "lumos_ct_010"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_146"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_168"),
    ("dataset/lumos_ct_211_280_dcm.zip", "lumos_ct_268"),
]


def main():

    results = []

    for zip_path, patient_folder in TESTS:

        volume, metadata = load_ct_volume(
            zip_path,
            patient_folder,
        )

        values = volume.reshape(-1)

        percentiles = np.percentile(
            values,
            [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5],
        )

        results.append(
            {
                "patient": patient_folder,
                "depth": volume.shape[0],
                "height": volume.shape[1],
                "width": volume.shape[2],
                "min": float(values.min()),
                "p0.5": float(percentiles[0]),
                "p1": float(percentiles[1]),
                "p5": float(percentiles[2]),
                "p25": float(percentiles[3]),
                "median": float(percentiles[4]),
                "p75": float(percentiles[5]),
                "p95": float(percentiles[6]),
                "p99": float(percentiles[7]),
                "p99.5": float(percentiles[8]),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )

    df = pd.DataFrame(results)

    print()
    print("=" * 100)
    print("CT HU DISTRIBUTION AUDIT")
    print("=" * 100)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()