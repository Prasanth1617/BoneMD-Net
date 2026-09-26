import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dataset.multimodal_dataset import LUMOSMultimodalDataset


MANIFEST_PATH = "results/patient_split.csv"
XRAY_ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
CT_BASE_DIR = "dataset"
CACHE_DIR = Path("cache") / "test"


def main():

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = LUMOSMultimodalDataset(
        MANIFEST_PATH,
        XRAY_ZIP_PATH,
        CT_BASE_DIR,
        split="test",
    )

    print("Split: test")
    print("Patients:", len(dataset))
    print()

    for index in range(len(dataset)):

        sample = dataset[index]

        patient_id = int(sample["patient_id"])

        output_path = (
            CACHE_DIR /
            f"patient_{patient_id:03d}.pt"
        )

        torch.save(
            {
                "ap": sample["ap"],
                "lateral": sample["lateral"],
                "ct": sample["ct"],
                "label": sample["label"],
                "patient_id": sample["patient_id"],
                "split": sample["split"],
            },
            output_path,
        )

        print(
            f"[{index + 1}/{len(dataset)}] "
            f"Patient {patient_id} cached "
            f"({output_path.stat().st_size / (1024 ** 2):.1f} MB)"
        )

    print()
    print("Test cache building completed successfully.")


if __name__ == "__main__":
    main()