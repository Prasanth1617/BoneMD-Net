import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


EXCEL_PATH = PROJECT_ROOT / "dataset" / "lumos_clinical_data.xlsx"


def main():
    df = pd.read_excel(EXCEL_PATH)

    print("=" * 80)
    print("LUMOS T-SCORE / LABEL INSPECTION")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print()

    print("Relevant columns:")
    for column in df.columns:
        if (
            "T_" in str(column)
            or "Z_" in str(column)
            or "Osteoporosis" in str(column)
        ):
            print(f"  {column}")

    print()
    print("Label counts:")
    print(df["Osteoporosis"].value_counts().sort_index())

    print()
    print("T-score summary:")
    tscore_columns = [
        column
        for column in df.columns
        if str(column).startswith("T_")
    ]

    print(
        df[tscore_columns].describe().T.to_string()
    )

    print()
    print("Mean T-scores by class:")

    class_means = (
        df.groupby("Osteoporosis")[tscore_columns]
        .mean()
        .T
    )

    print(class_means.to_string())

    print()
    print("Non-null T-score counts by class:")

    nonnull_counts = (
        df.groupby("Osteoporosis")[tscore_columns]
        .count()
        .T
    )

    print(nonnull_counts.to_string())

    print()
    print("First 20 patients:")
    display_columns = [
        "patient_id",
        "Osteoporosis",
    ] + tscore_columns

    print(
        df[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()