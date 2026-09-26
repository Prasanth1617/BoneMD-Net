import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.ct_loader import load_ct_volume

zip_path = "dataset/lumos_ct_141_210_dcm.zip"
patient_folder = "lumos_ct_175"

volume, metadata = load_ct_volume(zip_path, patient_folder)

native = metadata[0]["row_spacing"]
mid = volume.shape[0] // 2

resampled = zoom(
    volume,
    (1.0, native / 1.0, native / 1.0),
    order=1
)

h, w = resampled.shape[1:]
start_y = (h - 256) // 2
start_x = (w - 256) // 2

cropped = resampled[
    :,
    start_y:start_y + 256,
    start_x:start_x + 256
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(
    resampled[mid],
    cmap="gray",
    vmin=-300,
    vmax=1000
)
axes[0].set_title(f"1.0 mm\n{h} × {w}")
axes[0].axis("off")

axes[1].imshow(
    cropped[mid],
    cmap="gray",
    vmin=-300,
    vmax=1000
)
axes[1].set_title("Center crop\n256 × 256")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results/ct_256_crop_175.png", dpi=150)
plt.close()

print("CT 256 x 256 crop test")
print("-" * 50)
print(f"Native spacing: {native:.3f} mm")
print(f"Resampled volume: {resampled.shape}")
print(f"Cropped volume: {cropped.shape}")
print("Saved: results/ct_256_crop_175.png")
