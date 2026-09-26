from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "per_class_metrics.png"


def main():
    classes = [
        "Normal",
        "Osteopenia",
        "Osteoporosis",
    ]

    # Verified metrics from FINAL_BEST_TEACHER_75_61.pth
    precision = np.array([
        0.9286,
        0.6429,
        0.6923,
    ]) * 100

    recall = np.array([
        0.6842,
        0.7500,
        0.9000,
    ]) * 100

    f1 = np.array([
        0.7879,
        0.6923,
        0.7826,
    ]) * 100

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - width,
        precision,
        width,
        label="Precision",
    )

    plt.bar(
        x,
        recall,
        width,
        label="Recall",
    )

    plt.bar(
        x + width,
        f1,
        width,
        label="F1-score",
    )

    plt.ylabel("Score (%)")
    plt.xlabel("Class")
    plt.title("BoneMD-Net Teacher - Per-Class Test Performance")

    plt.xticks(
        x,
        classes,
    )

    plt.ylim(0, 100)
    plt.legend()

    for values, offsets in [
        (precision, -width),
        (recall, 0),
        (f1, width),
    ]:
        for i, value in enumerate(values):
            plt.text(
                i + offsets,
                value + 1.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()