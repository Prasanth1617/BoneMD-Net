import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.xray_loader import load_xray


MAPPING = "results/xray_image_mapping.csv"
ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
OUTPUT = "results/xray_aspect_extremes.png"


df = pd.read_csv(MAPPING)
df["aspect_ratio"] = df["columns"] / df["rows"]

# Select representative images.
narrow = df.loc[df["aspect_ratio"].idxmin()]
wide = df.loc[df["aspect_ratio"].idxmax()]
median_value = df["aspect_ratio"].median()
typical = df.iloc[(df["aspect_ratio"] - median_value).abs().argsort()[:1].iloc[0]]


samples = [
    ("Narrowest", narrow),
    ("Typical / Median", typical),
    ("Widest", wide),
]


fig, axes = plt.subplots(1, 3, figsize=(15, 7))

for ax, (name, row) in zip(axes, samples):
    image, ds = load_xray(ZIP_PATH, row["dicom_file"])

    # Robust display only — does NOT modify the loader or source data.
    lo, hi = np.percentile(image, [1, 99])
    display_image = np.clip((image - lo) / (hi - lo), 0, 1)

    ax.imshow(display_image, cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"{name}\n"
        f"Patient {int(row['patient_id'])} | {row['view']}\n"
        f"{int(row['rows'])} × {int(row['columns'])}\n"
        f"Aspect ratio: {row['aspect_ratio']:.3f}"
    )

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", OUTPUT)
print()
print("Selected images:")

for name, row in samples:
    print(
        f"{name}: "
        f"patient={int(row['patient_id'])}, "
        f"view={row['view']}, "
        f"shape={int(row['rows'])}x{int(row['columns'])}, "
        f"aspect_ratio={row['aspect_ratio']:.3f}, "
        f"file={row['dicom_file']}"
    )
