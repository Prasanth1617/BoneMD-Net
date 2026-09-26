import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.ct_loader import load_ct_volume


TESTS = [
    ("dataset/lumos_ct_001_070_dcm.zip", "lumos_ct_001"),
    ("dataset/lumos_ct_001_070_dcm.zip", "lumos_ct_010"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_146"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_168"),
    ("dataset/lumos_ct_211_280_dcm.zip", "lumos_ct_268"),
]


def main():

    print()
    print("=" * 100)
    print("CT BONE HU WINDOW AUDIT")
    print("=" * 100)

    for zip_path, patient_folder in TESTS:

        volume, metadata = load_ct_volume(
            zip_path,
            patient_folder,
        )

        values = volume.reshape(-1)

        print()
        print(patient_folder)
        print("-" * 60)

        for low, high in [
            (-200, 300),
            (-200, 500),
            (-100, 500),
            (-100, 1000),
            (-300, 1500),
        ]:

            mask = (values >= low) & (values <= high)

            percentage = 100.0 * mask.mean()

            print(
                f"Window [{low:5d}, {high:4d}] "
                f"-> {percentage:6.2f}% of voxels"
            )


if __name__ == "__main__":
    main()