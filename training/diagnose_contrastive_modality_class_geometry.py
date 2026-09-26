import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "teacher_contrastive_queue_best.pth"
TRAIN_CACHE = PROJECT_ROOT / "dataset" / "cache" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_centroid_distance(a, b):
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float(1.0 - torch.dot(a, b).item())


def analyze_modality(name, features, labels):
    print()
    print("=" * 70)
    print(f"{name} MODALITY GEOMETRY")
    print("=" * 70)

    features = F.normalize(features, p=2, dim=1)

    centroids = {}

    for cls in [0, 1, 2]:
        mask = labels == cls
        cls_features = features[mask]

        centroid = cls_features.mean(dim=0)
        centroid = F.normalize(centroid, dim=0)

        centroids[cls] = centroid

        print(
            f"Class {cls}: "
            f"n={int(mask.sum())}, "
            f"mean feature norm={cls_features.norm(dim=1).mean().item():.6f}"
        )

    print()
    print("Centroid cosine distances:")
    print(f"  Normal ↔ Osteopenia  : {cosine_centroid_distance(centroids[0], centroids[1]):.6f}")
    print(f"  Normal ↔ Osteoporosis : {cosine_centroid_distance(centroids[0], centroids[2]):.6f}")
    print(f"  Osteopenia ↔ Osteoporosis: {cosine_centroid_distance(centroids[1], centroids[2]):.6f}")

    # Nearest-centroid classification
    centroid_matrix = torch.stack(
        [centroids[0], centroids[1], centroids[2]], dim=0
    )

    similarities = features @ centroid_matrix.T
    predictions = similarities.argmax(dim=1)

    accuracy = (predictions == labels).float().mean().item()

    print()
    print(f"Nearest-centroid accuracy: {accuracy:.4f}")

    confusion = torch.zeros(3, 3, dtype=torch.int64)

    for true_label, pred_label in zip(labels, predictions):
        confusion[int(true_label), int(pred_label)] += 1

    print()
    print("Confusion matrix:")
    print(confusion.numpy())

    # Per-class recall
    print()
    print("Per-class recall:")

    for cls in [0, 1, 2]:
        total = confusion[cls].sum().item()
        correct = confusion[cls, cls].item()

        recall = correct / total if total > 0 else 0.0

        print(f"  Class {cls}: {recall:.4f}")

    return centroids


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

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint val accuracy: {checkpoint.get('val_acc')}")

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

            _, _, ap_feat, lateral_feat, ct_feat = model(
                ap,
                lateral,
                ct,
            )

            ap_features.append(ap_feat.squeeze(0).cpu())
            lateral_features.append(lateral_feat.squeeze(0).cpu())
            ct_features.append(ct_feat.squeeze(0).cpu())
            labels.append(int(sample["label"]))

    ap_features = torch.stack(ap_features)
    lateral_features = torch.stack(lateral_features)
    ct_features = torch.stack(ct_features)
    labels = torch.tensor(labels)

    print()
    print(f"Samples: {len(labels)}")
    print(
        "Class counts:",
        [
            int((labels == cls).sum())
            for cls in [0, 1, 2]
        ],
    )

    analyze_modality("AP", ap_features, labels)
    analyze_modality("LATERAL", lateral_features, labels)
    analyze_modality("CT", ct_features, labels)


if __name__ == "__main__":
    main()