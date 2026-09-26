import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from dataset.multimodal_dataset import LUMOSMultimodalDataset


MANIFEST_PATH = "results/patient_split.csv"
XRAY_ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
CT_BASE_DIR = "dataset"

CACHE_BASE_DIR = "cache"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    for split in ["train", "val"]:
        split_manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        cache_dir = Path(CACHE_BASE_DIR) / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = LUMOSMultimodalDataset(
            MANIFEST_PATH,
            XRAY_ZIP_PATH,
            CT_BASE_DIR,
            split=split,
        )

        print(f"Split: {split}")
        print(f"Patients: {len(dataset)}")

        for index in range(len(dataset)):
            sample = dataset[index]
            patient_id = int(sample["patient_id"])

            output_path = cache_dir / f"patient_{patient_id:03d}.pt"

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

    print("Cache building completed successfully.")


if __name__ == "__main__":
    main()
