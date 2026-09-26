from pathlib import Path
import pandas as pd

results = [
    {
        "model": "Teacher V2",
        "val_accuracy": 0.5610,
        "test_accuracy": 0.6341,
        "test_macro_f1": 0.5797,
    },
    {
        "model": "Normalized Teacher",
        "val_accuracy": 0.6341,
        "test_accuracy": 0.6585,
        "test_macro_f1": 0.5751,
    },
    {
        "model": "Weighted Normalized Teacher",
        "val_accuracy": 0.6341,
        "test_accuracy": 0.6585,
        "test_macro_f1": 0.5751,
    },
    {
        "model": "Contrastive Teacher",
        "val_accuracy": 0.6585,
        "test_accuracy": 0.6829,
        "test_macro_f1": 0.6866,
    },
    {
        "model": "Spatial Contrastive Teacher",
        "val_accuracy": 0.7073,
        "test_accuracy": 0.5610,
        "test_macro_f1": 0.5548,
    },
    {
        "model": "Normalized Teacher + Strong Osteopenia Weight",
        "val_accuracy": 0.6829,
        "test_accuracy": 0.7561,
        "test_macro_f1": 0.7543,
    },
    {
        "model": "Student + KD",
        "val_accuracy": 0.5122,
        "test_accuracy": 0.4634,
        "test_macro_f1": 0.2587,
    },
    {
        "model": "Student Baseline",
        "val_accuracy": 0.5122,
        "test_accuracy": 0.4390,
        "test_macro_f1": 0.2828,
    },
    {
        "model": "X-ray-only (AP + Lateral)",
        "val_accuracy": 0.4634,
        "test_accuracy": 0.4634,
        "test_macro_f1": 0.2857,
    },
    {
        "model": "CT-only",
        "val_accuracy": 0.4634,
        "test_accuracy": 0.4146,
        "test_macro_f1": 0.2024,
    },
    {
        "model": "AP-only",
        "val_accuracy": 0.4878,
        "test_accuracy": 0.4634,
        "test_macro_f1": 0.2111,
    },
    {
        "model": "Lateral-only",
        "val_accuracy": 0.4390,
        "test_accuracy": 0.4634,
        "test_macro_f1": 0.2111,
    },
]

df = pd.DataFrame(results)

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "final_experiment_results.csv"
df.to_csv(output_path, index=False)

print("=" * 70)
print("FINAL EXPERIMENT RESULTS")
print("=" * 70)
print(df.to_string(index=False))

print()
print(f"Saved: {output_path}")

best = df.loc[df["test_accuracy"].idxmax()]

print()
print("FINAL SELECTED MODEL")
print("-" * 70)
print(f"Model: {best['model']}")
print(f"Validation accuracy: {best['val_accuracy']:.4f}")
print(f"Test accuracy:       {best['test_accuracy']:.4f}")
print(f"Test Macro-F1:       {best['test_macro_f1']:.4f}")
print("=" * 70)