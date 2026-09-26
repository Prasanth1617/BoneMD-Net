import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.ct_loader import load_ct_volume

zip_path = "dataset/lumos_ct_141_210_dcm.zip"
patient_folder = "lumos_ct_175"

volume, metadata = load_ct_volume(zip_path, patient_folder)

native = metadata[0]["row_spacing"]
mid = volume.shape[0] // 2

resampled_05 = zoom(volume, (1, native / 0.5, native / 0.5), order=1)
resampled_10 = zoom(volume, (1, native / 1.0, native / 1.0), order=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(volume[mid], cmap="gray", vmin=-300, vmax=1000)
axes[0].set_title(f"Native\n{native:.3f} mm")
axes[0].axis("off")

axes[1].imshow(resampled_05[mid], cmap="gray", vmin=-300, vmax=1000)
axes[1].set_title(f"0.5 mm\n{resampled_05.shape[1]} × {resampled_05.shape[2]}")
axes[1].axis("off")

axes[2].imshow(resampled_10[mid], cmap="gray", vmin=-300, vmax=1000)
axes[2].set_title(f"1.0 mm\n{resampled_10.shape[1]} × {resampled_10.shape[2]}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("results/ct_inplane_compare_175.png", dpi=150)
plt.close()

print("CT in-plane comparison")
print("-" * 50)
print(f"Native: {volume.shape}")
print(f"Native spacing: {native:.3f} mm")
print(f"0.5 mm: {resampled_05.shape}")
print(f"1.0 mm: {resampled_10.shape}")
print("Saved: results/ct_inplane_compare_175.png")
