import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "teacher_contrastive_queue_best.pth"
TRAIN_CACHE = PROJECT_ROOT / "cache" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze(name, features, labels):
    print()
    print("=" * 70)
    print(f"{name} FEATURE VARIABILITY")
    print("=" * 70)

    features = F.normalize(features, p=2, dim=1)

    # Variation across patients for every feature dimension.
    feature_std = features.std(dim=0)

    print(f"Feature dimension: {features.shape[1]}")
    print(f"Mean feature std:   {feature_std.mean().item():.8f}")
    print(f"Median feature std: {feature_std.median().item():.8f}")
    print(f"Max feature std:    {feature_std.max().item():.8f}")
    print(f"Min feature std:    {feature_std.min().item():.8f}")

    # Pairwise cosine similarity.
    similarity = features @ features.T

    # Remove self-similarity.
    n = similarity.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)

    pairwise = similarity[mask]

    print()
    print("Pairwise cosine similarity:")
    print(f"Mean:   {pairwise.mean().item():.8f}")
    print(f"Std:    {pairwise.std().item():.8f}")
    print(f"Min:    {pairwise.min().item():.8f}")
    print(f"Max:    {pairwise.max().item():.8f}")

    # Within-class pairwise similarity.
    print()
    print("Within-class cosine similarity:")

    for cls in [0, 1, 2]:
        cls_features = features[labels == cls]

        cls_similarity = cls_features @ cls_features.T

        m = cls_similarity.shape[0]
        cls_mask = ~torch.eye(m, dtype=torch.bool)

        values = cls_similarity[cls_mask]

        print(
            f"Class {cls}: "
            f"mean={values.mean().item():.8f}, "
            f"std={values.std().item():.8f}, "
            f"min={values.min().item():.8f}, "
            f"max={values.max().item():.8f}"
        )

    # Mean pairwise distance.
    distance = 1.0 - pairwise

    print()
    print("Pairwise cosine distance:")
    print(f"Mean: {distance.mean().item():.8f}")
    print(f"Std:  {distance.std().item():.8f}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Cache: {TRAIN_CACHE}")

    dataset = CachedMultimodalDataset(str(TRAIN_CACHE))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherContrastive(
        num_classes=3,
        feature_dim=512,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")

    ap_features = []
    lateral_features = []
    ct_features = []
    labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            outputs = model(ap, lateral, ct)

            ap_features.append(
                outputs["ap_features"].squeeze(0).cpu()
            )

            lateral_features.append(
                outputs["lateral_features"].squeeze(0).cpu()
            )

            ct_features.append(
                outputs["ct_features"].squeeze(0).cpu()
            )

            labels.append(int(sample["label"]))

    ap_features = torch.stack(ap_features)
    lateral_features = torch.stack(lateral_features)
    ct_features = torch.stack(ct_features)
    labels = torch.tensor(labels)

    print()
    print(f"Samples: {len(labels)}")

    analyze("AP", ap_features, labels)
    analyze("LATERAL", lateral_features, labels)
    analyze("CT", ct_features, labels)


if __name__ == "__main__":
    main()