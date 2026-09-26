import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("results/confusion_matrices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matrices = {
    "teacher_v2": np.array([
        [17, 2, 0],
        [9, 2, 1],
        [1, 2, 7],
    ]),

    "student_kd": np.array([
        [18, 1, 0],
        [10, 1, 1],
        [9, 1, 0],
    ]),

    "student_baseline": np.array([
        [16, 0, 3],
        [12, 0, 0],
        [8, 0, 2],
    ]),

    "xray_only": np.array([
        [17, 2, 0],
        [10, 2, 0],
        [10, 0, 0],
    ]),

    "ct_only": np.array([
        [17, 2, 0],
        [12, 0, 0],
        [8, 2, 0],
    ]),

    "ap_only": np.array([
        [19, 0, 0],
        [12, 0, 0],
        [10, 0, 0],
    ]),

    "lateral_only": np.array([
        [19, 0, 0],
        [12, 0, 0],
        [10, 0, 0],
    ]),

    # FINAL MODEL
    "final_osteopenia_weighted": np.array([
        [13, 4, 2],
        [1, 9, 2],
        [0, 1, 9],
    ]),
}

labels = ["Normal", "Osteopenia", "Osteoporosis"]

for name, matrix in matrices.items():

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(matrix)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(name.replace("_", " ").title())

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Number of patients")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print()
print("All confusion matrices generated successfully.")