from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "results" / "final_experiment_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "final_model_comparison.png"

FINAL_MODEL_NAME = (
    "Normalized Teacher + Strong Osteopenia Weight "
    "+ Label Smoothing + Cosine LR"
)


def main():
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "model",
        "val_accuracy",
        "test_accuracy",
        "test_macro_f1",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print()
    print(df.to_string(index=False))
    print()

    # Convert to percentages
    plot_df = df.copy()
    plot_df["val_accuracy"] *= 100
    plot_df["test_accuracy"] *= 100
    plot_df["test_macro_f1"] *= 100

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(plot_df))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        plot_df["val_accuracy"],
        width=width,
        label="Validation Accuracy",
    )

    ax.bar(
        x,
        plot_df["test_accuracy"],
        width=width,
        label="Test Accuracy",
    )

    ax.bar(
        [i + width for i in x],
        plot_df["test_macro_f1"],
        width=width,
        label="Test Macro-F1",
    )

    ax.set_ylabel("Performance (%)")
    ax.set_xlabel("Model")
    ax.set_title("BoneMD-Net Model Performance Comparison")

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        plot_df["model"],
        rotation=45,
        ha="right",
    )

    ax.set_ylim(0, 100)
    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(OUTPUT_PATH)
    print()

    final_rows = plot_df[
        plot_df["model"] == FINAL_MODEL_NAME
    ]

    if final_rows.empty:
        raise ValueError(
            f"Final model not found in CSV: {FINAL_MODEL_NAME}"
        )

    final_row = final_rows.iloc[0]

    print("Final model:")
    print(f"  Validation Accuracy: {final_row['val_accuracy']:.2f}%")
    print(f"  Test Accuracy:       {final_row['test_accuracy']:.2f}%")
    print(f"  Test Macro-F1:       {final_row['test_macro_f1']:.2f}%")

    print()
    print("=" * 70)
    print("FIGURE GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()