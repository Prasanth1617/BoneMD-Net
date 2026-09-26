import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.ct_loader import load_ct_volume

zip_path = "dataset/lumos_ct_141_210_dcm.zip"
patient_folder = "lumos_ct_175"

volume, metadata = load_ct_volume(zip_path, patient_folder)

native_row = metadata[0]["row_spacing"]
native_col = metadata[0]["col_spacing"]

target_spacing = 0.5

zoom_y = native_row / target_spacing
zoom_x = native_col / target_spacing

resampled = zoom(
    volume,
    (1.0, zoom_y, zoom_x),
    order=1
)

mid = volume.shape[0] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(volume[mid], cmap="gray", vmin=-300, vmax=1000)
axes[0].set_title(
    f"Native\n{native_row:.3f} × {native_col:.3f} mm"
)
axes[0].axis("off")

axes[1].imshow(resampled[mid], cmap="gray", vmin=-300, vmax=1000)
axes[1].set_title(
    f"Resampled\n{target_spacing:.1f} × {target_spacing:.1f} mm"
)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results/ct_inplane_resample_175.png", dpi=150)
plt.close()

print("CT in-plane resampling visual test")
print("-" * 50)
print("Patient: 175")
print(f"Native volume: {volume.shape}")
print(f"Native spacing: {native_row:.3f} x {native_col:.3f} mm")
print(f"Target spacing: {target_spacing:.3f} x {target_spacing:.3f} mm")
print(f"Resampled volume: {resampled.shape}")
print("Saved: results/ct_inplane_resample_175.png")
