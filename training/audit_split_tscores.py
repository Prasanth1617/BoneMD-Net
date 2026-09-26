import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch


EXCEL_PATH = PROJECT_ROOT / "dataset" / "lumos_clinical_data.xlsx"

CACHE_DIRS = {
    "train": PROJECT_ROOT / "cache" / "train",
    "val": PROJECT_ROOT / "cache" / "val",
    "test": PROJECT_ROOT / "cache" / "test",
}


TSCORE_COLUMNS = [
    "T_L1-L2",
    "T_L1-L3",
    "T_L1-L4",
    "T_L2-L3",
    "T_L2-L4",
    "T_L3-L4",
]


def get_cache_patient_ids(cache_dir):
    patient_ids = []

    for path in sorted(cache_dir.glob("patient_*.pt")):
        sample = torch.load(
            path,
            map_location="cpu",
        )

        patient_ids.append(
            int(sample["patient_id"])
        )

    return patient_ids


def main():
    df = pd.read_excel(EXCEL_PATH)

    print("=" * 80)
    print("SPLIT-LEVEL T-SCORE AUDIT")
    print("=" * 80)

    clinical_ids = set(
        df["patient_id"].astype(int)
    )

    print(f"Clinical patients: {len(clinical_ids)}")

    all_split_ids = []

    for split_name, cache_dir in CACHE_DIRS.items():

        patient_ids = get_cache_patient_ids(
            cache_dir
        )

        all_split_ids.extend(patient_ids)

        print()
        print("-" * 80)
        print(f"{split_name.upper()} SPLIT")
        print("-" * 80)

        print(f"Cached patients: {len(patient_ids)}")
        print(f"Unique patients: {len(set(patient_ids))}")

        missing = [
            pid
            for pid in patient_ids
            if pid not in clinical_ids
        ]

        print(
            f"Missing clinical rows: {len(missing)}"
        )

        if missing:
            print(f"Missing IDs: {missing}")

        split_df = df[
            df["patient_id"].isin(patient_ids)
        ].copy()

        print(
            f"Matched clinical rows: {len(split_df)}"
        )

        print()
        print("Class counts:")

        print(
            split_df["Osteoporosis"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        split_df["mean_tscore"] = (
            split_df[TSCORE_COLUMNS]
            .mean(axis=1)
        )

        print()
        print("Mean T-score by class:")

        print(
            split_df
            .groupby("Osteoporosis")["mean_tscore"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .to_string()
        )

        print()
        print("Missing T-score values:")

        print(
            split_df[TSCORE_COLUMNS]
            .isna()
            .sum()
            .to_string()
        )

    print()
    print("=" * 80)
    print("GLOBAL SPLIT CHECK")
    print("=" * 80)

    print(
        f"Total cached patient entries: "
        f"{len(all_split_ids)}"
    )

    print(
        f"Unique cached patient IDs: "
        f"{len(set(all_split_ids))}"
    )

    duplicates = (
        pd.Series(all_split_ids)
        .value_counts()
    )

    duplicates = duplicates[
        duplicates > 1
    ]

    print(
        f"Patients appearing in multiple splits: "
        f"{len(duplicates)}"
    )

    if len(duplicates) > 0:
        print(duplicates.to_string())

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()