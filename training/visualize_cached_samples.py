import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import glob
import torch
import matplotlib.pyplot as plt


files = sorted(
    glob.glob("cache/train/patient_*.pt")
)[:12]

fig, axes = plt.subplots(
    6,
    2,
    figsize=(8, 18)
)

for row, path in enumerate(files[:6]):

    sample = torch.load(
        path,
        map_location="cpu"
    )

    patient_id = sample["patient_id"]
    label = sample["label"].item()

    ap = sample["ap"].squeeze().numpy()
    lateral = sample["lateral"].squeeze().numpy()

    axes[row, 0].imshow(ap, cmap="gray")
    axes[row, 0].set_title(
        f"Patient {patient_id} | Label {label} | AP"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(lateral, cmap="gray")
    axes[row, 1].set_title(
        f"Patient {patient_id} | Label {label} | Lateral"
    )
    axes[row, 1].axis("off")


plt.tight_layout()

output = "results/cached_xray_samples.png"

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

print("Saved:", output)