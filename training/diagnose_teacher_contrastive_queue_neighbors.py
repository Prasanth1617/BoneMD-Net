import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = PROJECT_ROOT / "cache" / "train"
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)


def main():
    print("Device:", DEVICE)

    dataset = CachedMultimodalDataset(CACHE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

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

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        checkpoint.get("val_accuracy"),
    )

    features = []
    labels = []
    patient_ids = []

    with torch.no_grad():
        for batch in loader:
            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(ap, lateral, ct)

            fused = output["fused_features"][0]

            features.append(
                fused.cpu().numpy()
            )

            labels.append(
                int(batch["label"][0])
            )

            patient_ids.append(
                int(batch["patient_id"][0])
            )

    features = np.asarray(features)
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)

    print()
    print("=" * 70)
    print("CROSS-CLASS NEAREST-NEIGHBOR DIAGNOSTIC")
    print("=" * 70)

    print("Samples:", len(features))
    print("Feature dimension:", features.shape[1])

    # Pairwise Euclidean distances.
    diff = features[:, None, :] - features[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Ignore self-distance.
    np.fill_diagonal(distances, np.inf)

    class_names = {
        0: "Normal",
        1: "Osteopenia",
        2: "Osteoporosis",
    }

    for source_class, target_class in [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]:
        print()
        print(
            f"{class_names[source_class]} -> "
            f"{class_names[target_class]}"
        )
        print("-" * 70)

        source_indices = np.where(
            labels == source_class
        )[0]

        target_indices = np.where(
            labels == target_class
        )[0]

        pairs = []

        for source_idx in source_indices:
            target_distances = distances[
                source_idx,
                target_indices,
            ]

            nearest_position = np.argmin(
                target_distances
            )

            target_idx = target_indices[
                nearest_position
            ]

            distance = distances[
                source_idx,
                target_idx
            ]

            pairs.append(
                (
                    distance,
                    patient_ids[source_idx],
                    patient_ids[target_idx],
                )
            )

        pairs.sort(key=lambda x: x[0])

        for distance, source_patient, target_patient in pairs[:10]:
            print(
                f"{source_patient:03d} -> "
                f"{target_patient:03d}   "
                f"distance={distance:.6f}"
            )

    print()
    print("=" * 70)
    print("GLOBAL CLOSEST CROSS-CLASS PAIRS")
    print("=" * 70)

    all_pairs = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            if labels[i] == labels[j]:
                continue

            all_pairs.append(
                (
                    distances[i, j],
                    patient_ids[i],
                    int(labels[i]),
                    patient_ids[j],
                    int(labels[j]),
                )
            )

    all_pairs.sort(key=lambda x: x[0])

    print()
    print(
        "Distance     Patient A   Class A       "
        "Patient B   Class B"
    )
    print("-" * 70)

    for distance, patient_a, class_a, patient_b, class_b in all_pairs[:30]:
        print(
            f"{distance:10.6f}     "
            f"{patient_a:03d}        "
            f"{class_names[class_a]:12s} "
            f"{patient_b:03d}        "
            f"{class_names[class_b]}"
        )

    print()
    print("=" * 70)
    print("CLOSEST OSTEOPENIA <-> OSTEOPOROSIS")
    print("=" * 70)

    disease_pairs = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            classes = {int(labels[i]), int(labels[j])}

            if classes != {1, 2}:
                continue

            disease_pairs.append(
                (
                    distances[i, j],
                    patient_ids[i],
                    int(labels[i]),
                    patient_ids[j],
                    int(labels[j]),
                )
            )

    disease_pairs.sort(key=lambda x: x[0])

    for distance, patient_a, class_a, patient_b, class_b in disease_pairs[:20]:
        print(
            f"{patient_a:03d} "
            f"({class_names[class_a]}) <-> "
            f"{patient_b:03d} "
            f"({class_names[class_b]}) "
            f"distance={distance:.6f}"
        )

    print()
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()