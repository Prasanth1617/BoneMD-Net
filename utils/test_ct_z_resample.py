import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.ct_loader import load_ct_volume

zip_path = "dataset/lumos_ct_141_210_dcm.zip"
patient_folder = "lumos_ct_175"

volume, metadata = load_ct_volume(zip_path, patient_folder)

native_spacing = abs(metadata[1]["z"] - metadata[0]["z"])
target_spacing = 2.5

zoom_factor = native_spacing / target_spacing
new_depth = round(volume.shape[0] * zoom_factor)

resampled = zoom(
    volume,
    (zoom_factor, 1, 1),
    order=1
)

mid_native = volume.shape[0] // 2
mid_resampled = resampled.shape[0] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(volume[mid_native], cmap="gray", vmin=-300, vmax=1000)
axes[0].set_title(
    f"Native\n{volume.shape[0]} slices, {native_spacing:.1f} mm"
)
axes[0].axis("off")

axes[1].imshow(resampled[mid_resampled], cmap="gray", vmin=-300, vmax=1000)
axes[1].set_title(
    f"Resampled\n{resampled.shape[0]} slices, {target_spacing:.1f} mm"
)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results/ct_z_resample_175.png", dpi=150)
plt.close()

print("CT Z-resampling visual test")
print("-" * 50)
print("Patient: 175")
print(f"Native volume: {volume.shape}")
print(f"Native Z spacing: {native_spacing:.3f} mm")
print(f"Target Z spacing: {target_spacing:.3f} mm")
print(f"Expected depth: {new_depth}")
print(f"Resampled volume: {resampled.shape}")
print("Saved: results/ct_z_resample_175.png")
