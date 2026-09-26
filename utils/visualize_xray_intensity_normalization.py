import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.xray_loader import load_xray


MANIFEST = "results/final_multimodal_manifest.csv"
MAPPING = "results/xray_image_mapping.csv"
ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
OUTPUT = "results/xray_intensity_normalization.png"


manifest = pd.read_csv(MANIFEST)
mapping = pd.read_csv(MAPPING)

records = []

for _, row in manifest.iterrows():
    records.append({
        "patient_id": int(row["patient_id"]),
        "view": "AP",
        "dicom_file": row["ap_dicom_file"],
    })
    records.append({
        "patient_id": int(row["patient_id"]),
        "view": "Lateral",
        "dicom_file": row["lateral_dicom_file"],
    })

records = pd.DataFrame(records)

# Select representative images:
# lowest P99, median P99, highest P99
stats = []

for _, row in records.iterrows():
    image, ds = load_xray(ZIP_PATH, row["dicom_file"])

    p1, p99 = np.percentile(image, [1, 99])

    stats.append({
        "patient_id": row["patient_id"],
        "view": row["view"],
        "dicom_file": row["dicom_file"],
        "p1": p1,
        "p99": p99,
    })

stats = pd.DataFrame(stats)

low = stats.loc[stats["p99"].idxmin()]
high = stats.loc[stats["p99"].idxmax()]

median_p99 = stats["p99"].median()
typical = stats.iloc[
    (stats["p99"] - median_p99).abs().argsort()[:1].iloc[0]
]

samples = [
    ("Lowest P99", low),
    ("Typical / Median P99", typical),
    ("Highest P99", high),
]


def percentile_normalize(image):
    p1, p99 = np.percentile(image, [1, 99])

    if p99 <= p1:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - p1) / (p99 - p1)

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for col, (name, row) in enumerate(samples):

    image, ds = load_xray(
        ZIP_PATH,
        row["dicom_file"],
    )

    normalized = percentile_normalize(image)

    axes[0, col].imshow(image, cmap="gray")
    axes[0, col].axis("off")
    axes[0, col].set_title(
        f"{name}\n"
        f"Patient {int(row['patient_id'])} | {row['view']}\n"
        f"P99 = {row['p99']:.0f}"
    )

    axes[1, col].imshow(normalized, cmap="gray")
    axes[1, col].axis("off")
    axes[1, col].set_title(
        "1–99 percentile normalized"
    )


plt.tight_layout()
plt.savefig(
    OUTPUT,
    dpi=150,
    bbox_inches="tight",
)
plt.close()


print("Saved:", OUTPUT)
print()
print("Selected images:")

for name, row in samples:
    print(
        f"{name}: "
        f"patient={int(row['patient_id'])}, "
        f"view={row['view']}, "
        f"P1={row['p1']:.1f}, "
        f"P99={row['p99']:.1f}, "
        f"file={row['dicom_file']}"
    )
