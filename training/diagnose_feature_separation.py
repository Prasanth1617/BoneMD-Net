import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_features(model, dataset):
    features = []
    labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            output = model(ap, lateral, ct)

            features.append(output["fused_features"].cpu())
            labels.append(sample["label"].reshape(1))

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def nearest_centroid_accuracy(features, labels, centroids):
    distances = torch.cdist(features, centroids)
    predictions = distances.argmin(dim=1)

    accuracy = (predictions == labels).float().mean()

    return accuracy, predictions


def main():
    print("Device:", DEVICE)

    checkpoint = torch.load(
        "checkpoints/teacher_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherV1().to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    train_dataset = CachedMultimodalDataset("cache/train")
    val_dataset = CachedMultimodalDataset("cache/val")

    train_features, train_labels = collect_features(
        model,
        train_dataset,
    )

    val_features, val_labels = collect_features(
        model,
        val_dataset,
    )

    print("\nTrain features:", tuple(train_features.shape))
    print("Val features:  ", tuple(val_features.shape))

    # ---------------------------------------------------------
    # Build class centroids from TRAIN features only
    # ---------------------------------------------------------

    centroids = []

    print("\nCLASS CENTROIDS")
    print("-" * 70)

    for label in [0, 1, 2]:
        mask = train_labels == label
        centroid = train_features[mask].mean(dim=0)
        centroids.append(centroid)

        print(
            f"Class {label}: "
            f"N={int(mask.sum())} | "
            f"centroid norm={torch.norm(centroid):.6f}"
        )

    centroids = torch.stack(centroids)

    # ---------------------------------------------------------
    # Train nearest-centroid accuracy
    # ---------------------------------------------------------

    train_accuracy, train_predictions = nearest_centroid_accuracy(
        train_features,
        train_labels,
        centroids,
    )

    val_accuracy, val_predictions = nearest_centroid_accuracy(
        val_features,
        val_labels,
        centroids,
    )

    print("\n")
    print("=" * 70)
    print("NEAREST-CENTROID RESULTS")
    print("=" * 70)

    print(
        "Train accuracy:",
        f"{float(train_accuracy) * 100:.2f}%"
    )

    print(
        "Val accuracy:",
        f"{float(val_accuracy) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------

    train_cm = torch.zeros(3, 3, dtype=torch.long)
    val_cm = torch.zeros(3, 3, dtype=torch.long)

    for true, pred in zip(train_labels, train_predictions):
        train_cm[int(true), int(pred)] += 1

    for true, pred in zip(val_labels, val_predictions):
        val_cm[int(true), int(pred)] += 1

    print("\nTrain nearest-centroid confusion matrix:")
    print(train_cm)

    print("\nValidation nearest-centroid confusion matrix:")
    print(val_cm)

    # ---------------------------------------------------------
    # Pairwise centroid distances
    # ---------------------------------------------------------

    centroid_distances = torch.cdist(
        centroids,
        centroids,
    )

    print("\n")
    print("=" * 70)
    print("CENTROID DISTANCES")
    print("=" * 70)

    print(centroid_distances)

    # ---------------------------------------------------------
    # Within-class feature spread
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WITHIN-CLASS SPREAD")
    print("=" * 70)

    for label in [0, 1, 2]:
        mask = train_labels == label
        class_features = train_features[mask]
        centroid = centroids[label]

        distances = torch.norm(
            class_features - centroid,
            dim=1,
        )

        print(
            f"Class {label}: "
            f"mean distance={distances.mean():.6f} | "
            f"std={distances.std():.6f}"
        )

    print("\n")
    print("=" * 70)
    print("FEATURE SEPARATION DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()