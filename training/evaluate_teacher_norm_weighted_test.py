import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_norm import BoneMDTeacherNorm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "FINAL_BEST_TEACHER_75_61.pth"
)

TEST_CACHE = PROJECT_ROOT / "cache" / "test"

CLASS_NAMES = [
    "Normal",
    "Osteopenia",
    "Osteoporosis",
]


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    dataset = CachedMultimodalDataset(TEST_CACHE)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    print(f"Test samples: {len(dataset)}")

    model = BoneMDTeacherNorm(
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

    all_labels = []
    all_predictions = []
    patient_results = []

    with torch.no_grad():
        for batch in loader:
            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = model(
                ap,
                lateral,
                ct,
            )

            predictions = torch.argmax(
                output["logits"],
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            patient_id = int(
                batch["patient_id"][0]
            )

            patient_results.append(
                (
                    patient_id,
                    int(labels.item()),
                    int(predictions.item()),
                )
            )

    accuracy = (
        sum(
            y_true == y_pred
            for y_true, y_pred
            in zip(all_labels, all_predictions)
        )
        / len(all_labels)
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
    )

    print()
    print("=" * 70)
    print("WEIGHTED NORMALIZED TEACHER — TEST EVALUATION")
    print("=" * 70)

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(
        f"Best validation accuracy: "
        f"{checkpoint.get('val_accuracy', 'unknown')}"
    )

    print()
    print(f"TEST ACCURACY: {accuracy:.4f}")

    print()
    print("Confusion matrix:")
    print(cm)

    print()
    print("Classification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print()
    print("Predicted class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = all_predictions.count(class_id)
        print(
            f"  {class_id} ({class_name}): {count}"
        )

    print()
    print("Patient-level predictions:")

    for patient_id, true_label, pred_label in patient_results:
        print(
            f"Patient {patient_id:03d}: "
            f"true={true_label} ({CLASS_NAMES[true_label]}) "
            f"pred={pred_label} ({CLASS_NAMES[pred_label]})"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
