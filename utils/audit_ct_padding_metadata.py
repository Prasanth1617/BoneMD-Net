import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zipfile
import numpy as np
import pydicom


TESTS = [
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_146"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_168"),
]


def main():

    print()
    print("=" * 100)
    print("CT PADDING / RAW PIXEL METADATA AUDIT")
    print("=" * 100)

    for zip_path, patient_folder in TESTS:

        with zipfile.ZipFile(zip_path, "r") as archive:

            names = [
                name
                for name in archive.namelist()
                if name.startswith(patient_folder + "/")
                and name.lower().endswith((".dcm", ".dicom"))
            ]

            # Inspect the first DICOM file.
            name = names[0]

            with archive.open(name) as f:
                ds = pydicom.dcmread(f)

            raw = ds.pixel_array

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))

            hu = raw.astype(np.float32) * slope + intercept

            print()
            print(patient_folder)
            print("-" * 60)
            print("File:", name)
            print("Rows:", ds.Rows)
            print("Columns:", ds.Columns)
            print("BitsAllocated:", ds.BitsAllocated)
            print("BitsStored:", ds.BitsStored)
            print("PixelRepresentation:", ds.PixelRepresentation)
            print("RescaleSlope:", slope)
            print("RescaleIntercept:", intercept)
            print("Raw dtype:", raw.dtype)
            print("Raw minimum:", raw.min())
            print("Raw maximum:", raw.max())

            unique, counts = np.unique(raw, return_counts=True)

            print()
            print("Most frequent raw pixel values:")

            order = np.argsort(counts)[::-1][:10]

            for index in order:
                raw_value = unique[index]
                count = counts[index]
                hu_value = raw_value * slope + intercept

                print(
                    f"raw={raw_value:6d} "
                    f"HU={hu_value:8.1f} "
                    f"count={count:10d} "
                    f"({100.0 * count / raw.size:6.2f}%)"
                )


if __name__ == "__main__":
    main()