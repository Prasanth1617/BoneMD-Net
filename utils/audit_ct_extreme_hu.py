import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.ct_loader import load_ct_volume


TESTS = [
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_146"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_168"),
]


def main():

    print()
    print("=" * 100)
    print("CT EXTREME HU AUDIT")
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

        for threshold in [-4000, -3000, -2500, -2000, -1500, -1200, -1000]:

            count = np.sum(values < threshold)
            percentage = 100.0 * count / values.size

            print(
                f"HU < {threshold:5d}: "
                f"{count:10d} voxels "
                f"({percentage:7.3f}%)"
            )

        print()
        print("Lowest unique HU values:")

        unique, counts = np.unique(values, return_counts=True)

        for hu, count in zip(unique[:20], counts[:20]):

            print(
                f"{hu:8.1f} HU -> {count:10d} voxels"
            )


if __name__ == "__main__":
    main()