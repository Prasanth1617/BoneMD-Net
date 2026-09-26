import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from utils.xray_loader import load_xray


MAPPING = "results/xray_image_mapping.csv"
ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
OUTPUT = "results/xray_resize_pad_224.png"

TARGET_SIZE = 224


def resize_and_pad(image, target_size=224):
    h, w = image.shape

    scale = min(target_size / h, target_size / w)

    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    padded = np.zeros((target_size, target_size), dtype=np.float32)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    padded[top:top + new_h, left:left + new_w] = resized

    return padded, (new_h, new_w), (top, left)


df = pd.read_csv(MAPPING)
df["aspect_ratio"] = df["columns"] / df["rows"]

narrow = df.loc[df["aspect_ratio"].idxmin()]
wide = df.loc[df["aspect_ratio"].idxmax()]

median_value = df["aspect_ratio"].median()
typical = df.iloc[
    (df["aspect_ratio"] - median_value).abs().argsort()[:1].iloc[0]
]

samples = [
    ("Narrowest", narrow),
    ("Typical / Median", typical),
    ("Widest", wide),
]


fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for col, (name, row) in enumerate(samples):

    image, ds = load_xray(
        ZIP_PATH,
        row["dicom_file"],
    )

    lo, hi = np.percentile(image, [1, 99])
    display_image = np.clip(
        (image - lo) / (hi - lo),
        0,
        1,
    )

    processed, new_shape, padding = resize_and_pad(
        display_image,
        TARGET_SIZE,
    )

    axes[0, col].imshow(display_image, cmap="gray")
    axes[0, col].axis("off")
    axes[0, col].set_title(
        f"{name}\n"
        f"{int(row['rows'])} × {int(row['columns'])}\n"
        f"AR: {row['aspect_ratio']:.3f}"
    )

    axes[1, col].imshow(processed, cmap="gray")
    axes[1, col].axis("off")
    axes[1, col].set_title(
        f"224 × 224\n"
        f"resized: {new_shape[0]} × {new_shape[1]}"
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
print("Preprocessing results:")

for name, row in samples:

    image, ds = load_xray(
        ZIP_PATH,
        row["dicom_file"],
    )

    processed, new_shape, padding = resize_and_pad(
        image,
        TARGET_SIZE,
    )

    print(
        f"{name}: "
        f"original={image.shape}, "
        f"resized={new_shape}, "
        f"padding={padding}, "
        f"final={processed.shape}"
    )
