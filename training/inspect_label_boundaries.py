import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


EXCEL_PATH = PROJECT_ROOT / "dataset" / "lumos_clinical_data.xlsx"


def main():
    df = pd.read_excel(EXCEL_PATH)

    tscore_columns = [
        "T_L1-L2",
        "T_L1-L3",
        "T_L1-L4",
        "T_L2-L3",
        "T_L2-L4",
        "T_L3-L4",
    ]

    df["min_tscore"] = df[tscore_columns].min(axis=1)
    df["mean_tscore"] = df[tscore_columns].mean(axis=1)

    print("=" * 80)
    print("LUMOS T-SCORE SEVERITY ANALYSIS")
    print("=" * 80)

    print("\nMinimum T-score by class:")
    print(
        df.groupby("Osteoporosis")["min_tscore"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .to_string()
    )

    print("\nMean T-score by class:")
    print(
        df.groupby("Osteoporosis")["mean_tscore"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .to_string()
    )

    print("\nMinimum T-score range by class:")

    for label in [0, 1, 2]:
        values = df.loc[
            df["Osteoporosis"] == label,
            "min_tscore"
        ]

        print(
            f"Class {label}: "
            f"min={values.min():.3f}, "
            f"25%={values.quantile(.25):.3f}, "
            f"50%={values.median():.3f}, "
            f"75%={values.quantile(.75):.3f}, "
            f"max={values.max():.3f}"
        )

    print("\nPatients whose minimum T-score crosses the class thresholds:")

    for label in [0, 1, 2]:
        subset = df[df["Osteoporosis"] == label]

        if label == 0:
            threshold_count = (
                subset["min_tscore"] < -1.0
            )
            description = "Normal with min T-score < -1.0"

        elif label == 1:
            threshold_count = (
                (subset["min_tscore"] <= -2.5)
            )
            description = "Osteopenia with min T-score <= -2.5"

        else:
            threshold_count = (
                subset["min_tscore"] > -2.5
            )
            description = "Osteoporosis with min T-score > -2.5"

        print(
            f"  {description}: "
            f"{int(threshold_count.sum())}"
        )

    print("\nFirst 20 patients:")
    print(
        df[
            [
                "patient_id",
                "Osteoporosis",
                "min_tscore",
                "mean_tscore",
            ]
            + tscore_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()