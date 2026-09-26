import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best_v1.pth"
)

CLASS_NAMES = [
    "Normal",
    "Osteopenia",
    "Osteoporosis",
]


def main():

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "test"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Test samples:", len(dataset))

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
        checkpoint["val_accuracy"],
    )
    print()

    true_labels = []
    predictions = []
    patient_ids = []

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

            predicted = output["logits"].argmax(
                dim=1
            )

            true_labels.append(
                int(labels.item())
            )

            predictions.append(
                int(predicted.item())
            )

            patient_ids.append(
                int(batch["patient_id"].item())
            )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2],
    )

    print("=" * 70)
    print("NORMALIZED TEACHER TEST RESULTS")
    print("=" * 70)

    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    print()
    print("Classification report:")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Predicted class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = predictions.count(class_id)

        print(
            f"  {class_id} ({class_name}): {count}"
        )

    print()
    print("Test patient predictions:")

    for patient_id, true_label, prediction in zip(
        patient_ids,
        true_labels,
        predictions,
    ):

        print(
            f"Patient {patient_id:03d} | "
            f"True: {CLASS_NAMES[true_label]} | "
            f"Predicted: {CLASS_NAMES[prediction]}"
        )


if __name__ == "__main__":
    main()
