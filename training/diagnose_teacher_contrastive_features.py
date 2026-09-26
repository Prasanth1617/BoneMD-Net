import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)


def collect_features(model, loader):

    model.eval()

    features = []
    labels = []
    patient_ids = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(ap, lateral, ct)

            features.append(
                output["fused_features"].cpu()
            )

            labels.append(
                batch["label"].cpu()
            )

            patient_ids.append(
                batch["patient_id"].cpu()
            )

    return (
        torch.cat(features),
        torch.cat(labels),
        torch.cat(patient_ids),
    )


def pairwise_distances(features):

    return torch.cdist(
        features,
        features,
        p=2,
    )


def main():

    print("Device:", DEVICE)

    model = BoneMDTeacherContrastive(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"],
    )
    print()

    train_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "train"
    )

    val_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    train_features, train_labels, train_ids = (
        collect_features(model, train_loader)
    )

    val_features, val_labels, val_ids = (
        collect_features(model, val_loader)
    )

    print("Train features:", tuple(train_features.shape))
    print("Val features:", tuple(val_features.shape))
    print()

    # ------------------------------------------------------------
    # Feature norms
    # ------------------------------------------------------------

    train_norms = torch.linalg.vector_norm(
        train_features,
        dim=1,
    )

    val_norms = torch.linalg.vector_norm(
        val_features,
        dim=1,
    )

    print("FUSED FEATURE NORMS")
    print("-" * 60)

    print(
        "Train mean:",
        train_norms.mean().item(),
    )

    print(
        "Train std:",
        train_norms.std().item(),
    )

    print(
        "Val mean:",
        val_norms.mean().item(),
    )

    print(
        "Val std:",
        val_norms.std().item(),
    )

    print()

    # ------------------------------------------------------------
    # Class centroids
    # ------------------------------------------------------------

    centroids = {}

    for class_id in [0, 1, 2]:

        mask = train_labels == class_id

        centroid = train_features[mask].mean(
            dim=0
        )

        centroids[class_id] = centroid

        print(
            f"Class {class_id} centroid norm:",
            torch.linalg.vector_norm(
                centroid
            ).item(),
        )

    print()

    # ------------------------------------------------------------
    # Centroid distances
    # ------------------------------------------------------------

    print("CENTROID DISTANCES")
    print("-" * 60)

    for a, b in [(0, 1), (0, 2), (1, 2)]:

        distance = torch.linalg.vector_norm(
            centroids[a] - centroids[b]
        )

        print(
            f"{a} <-> {b}:",
            distance.item(),
        )

    print()

    # ------------------------------------------------------------
    # Nearest centroid classification
    # ------------------------------------------------------------

    centroid_matrix = torch.stack(
        [
            centroids[0],
            centroids[1],
            centroids[2],
        ]
    )

    def nearest_centroid_accuracy(
        features,
        labels,
    ):

        distances = torch.cdist(
            features,
            centroid_matrix,
            p=2,
        )

        predictions = distances.argmin(
            dim=1
        )

        accuracy = (
            predictions == labels
        ).float().mean().item()

        return accuracy, predictions

    train_acc, train_pred = (
        nearest_centroid_accuracy(
            train_features,
            train_labels,
        )
    )

    val_acc, val_pred = (
        nearest_centroid_accuracy(
            val_features,
            val_labels,
        )
    )

    print("NEAREST-CENTROID RESULTS")
    print("-" * 60)

    print(
        f"Train accuracy: {train_acc:.4f}"
    )

    print(
        f"Val accuracy:   {val_acc:.4f}"
    )

    print()

    print("Train confusion:")
    print(
        confusion_matrix(
            train_labels.numpy(),
            train_pred.numpy(),
            labels=[0, 1, 2],
        )
    )

    print()

    print("Val confusion:")
    print(
        confusion_matrix(
            val_labels.numpy(),
            val_pred.numpy(),
            labels=[0, 1, 2],
        )
    )

    print()

    # ------------------------------------------------------------
    # Within-class spread
    # ------------------------------------------------------------

    print("WITHIN-CLASS FEATURE SPREAD")
    print("-" * 60)

    train_distances = pairwise_distances(
        train_features
    )

    for class_id in [0, 1, 2]:

        mask = train_labels == class_id
        indices = torch.where(mask)[0]

        class_distances = train_distances[
            indices[:, None],
            indices[None, :],
        ]

        upper = class_distances[
            torch.triu(
                torch.ones_like(
                    class_distances,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        ]

        print(
            f"Class {class_id} mean distance:",
            upper.mean().item(),
        )

    print()

    # ------------------------------------------------------------
    # Modality feature norms
    # ------------------------------------------------------------

    print("NORMALIZED MODALITY FEATURE NORMS")
    print("-" * 60)

    model.eval()

    ap_norms = []
    lateral_norms = []
    ct_norms = []

    with torch.no_grad():

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(ap, lateral, ct)

            ap_norms.append(
                torch.linalg.vector_norm(
                    output["ap_features"],
                    dim=1,
                ).cpu()
            )

            lateral_norms.append(
                torch.linalg.vector_norm(
                    output["lateral_features"],
                    dim=1,
                ).cpu()
            )

            ct_norms.append(
                torch.linalg.vector_norm(
                    output["ct_features"],
                    dim=1,
                ).cpu()
            )

    ap_norms = torch.cat(ap_norms)
    lateral_norms = torch.cat(lateral_norms)
    ct_norms = torch.cat(ct_norms)

    print(
        "AP mean norm:",
        ap_norms.mean().item(),
    )

    print(
        "Lateral mean norm:",
        lateral_norms.mean().item(),
    )

    print(
        "CT mean norm:",
        ct_norms.mean().item(),
    )


if __name__ == "__main__":
    main()
