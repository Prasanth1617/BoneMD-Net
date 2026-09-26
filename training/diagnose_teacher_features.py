import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/teacher_best_v2.pth"

dataset = CachedMultimodalDataset("cache/train")

model = BoneMDTeacher(
    feature_dim=512,
    num_classes=3
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

ap_features = []
lateral_features = []
ct_features = []
fused_features = []
labels = []

with torch.no_grad():

    for i in range(len(dataset)):

        sample = dataset[i]

        ap = sample["ap"].unsqueeze(0).to(DEVICE)
        lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
        ct = sample["ct"].unsqueeze(0).to(DEVICE)

        output = model(ap, lateral, ct)

        ap_features.append(output["ap_features"].squeeze(0).cpu())
        lateral_features.append(output["lateral_features"].squeeze(0).cpu())
        ct_features.append(output["ct_features"].squeeze(0).cpu())
        fused_features.append(output["fused_features"].squeeze(0).cpu())
        labels.append(sample["label"].item())


ap_features = torch.stack(ap_features)
lateral_features = torch.stack(lateral_features)
ct_features = torch.stack(ct_features)
fused_features = torch.stack(fused_features)
labels = torch.tensor(labels)


def report(name, features):

    print(f"\n{name}")
    print("-" * 60)

    print("Shape:", tuple(features.shape))

    print(
        "Overall mean: "
        f"{features.mean().item():.8f}"
    )

    print(
        "Overall std:  "
        f"{features.std().item():.8f}"
    )

    print(
        "Overall min:  "
        f"{features.min().item():.8f}"
    )

    print(
        "Overall max:  "
        f"{features.max().item():.8f}"
    )

    # Variation between patients
    per_dimension_std = features.std(dim=0)

    print(
        "Mean feature-dimension std: "
        f"{per_dimension_std.mean().item():.8f}"
    )

    print(
        "Max feature-dimension std:  "
        f"{per_dimension_std.max().item():.8f}"
    )

    # Average feature vector norm
    norms = torch.norm(features, dim=1)

    print(
        "Mean patient feature norm: "
        f"{norms.mean().item():.8f}"
    )

    print(
        "Std patient feature norm:  "
        f"{norms.std().item():.8f}"
    )


print("Device:", DEVICE)
print("Samples:", len(dataset))

report("AP FEATURES", ap_features)
report("LATERAL FEATURES", lateral_features)
report("CT FEATURES", ct_features)
report("FUSED FEATURES", fused_features)


print("\n\nFIRST 10 PATIENT FEATURE NORMS")
print("=" * 60)

for i in range(min(10, len(dataset))):

    print(
        f"Patient {i:3d} | "
        f"Label {labels[i].item()} | "
        f"AP={torch.norm(ap_features[i]).item():.6f} | "
        f"Lateral={torch.norm(lateral_features[i]).item():.6f} | "
        f"CT={torch.norm(ct_features[i]).item():.6f} | "
        f"Fused={torch.norm(fused_features[i]).item():.6f}"
    )