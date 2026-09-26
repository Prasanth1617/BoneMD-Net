import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "results/final_experiment_results.csv"
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

x = range(len(df))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    [i - width for i in x],
    df["val_accuracy"] * 100,
    width,
    label="Validation Accuracy",
)

ax.bar(
    x,
    df["test_accuracy"] * 100,
    width,
    label="Test Accuracy",
)

ax.bar(
    [i + width for i in x],
    df["test_macro_f1"] * 100,
    width,
    label="Test Macro F1",
)

ax.set_ylabel("Score (%)")
ax.set_title("BoneMD-Net Experimental Performance Comparison")
ax.set_xticks(list(x))
ax.set_xticklabels(
    [
        "Teacher V2",
        "Student + KD",
        "Student Baseline",
        "X-ray Only",
        "CT Only",
        "AP Only",
        "Lateral Only",
    ],
    rotation=25,
    ha="right",
)

ax.set_ylim(0, 70)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

output = OUTPUT_DIR / "model_performance_comparison.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output}")
print(f"Rows plotted: {len(df)}")
print("Performance comparison figure generated successfully.")
