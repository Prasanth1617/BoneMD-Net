from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "final_confusion_matrix.png"


def main():
    cm = np.array([
        [13, 6, 0],
        [4, 6, 2],
        [1, 0, 9],
    ])

    class_names = [
        "Normal",
        "Osteopenia",
        "Osteoporosis",
    ]

    plt.figure(figsize=(7, 6))

    plt.imshow(cm, interpolation="nearest")

    plt.title("BoneMD-Net Teacher — Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=30,
        ha="right"
    )

    plt.yticks(
        np.arange(len(class_names)),
        class_names
    )

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar(label="Number of patients")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()