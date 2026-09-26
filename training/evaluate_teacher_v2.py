import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/teacher_v2_controlled_best.pth"
DATA_CACHE = "cache/test"


def main():

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("Data cache:", DATA_CACHE)
    print()

    dataset = CachedMultimodalDataset(DATA_CACHE)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    teacher = BoneMDTeacher(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"],
    )
    print()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = teacher(
                ap,
                lateral,
                ct,
            )

            predictions = output["logits"].argmax(dim=1)

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
    )

    print("=" * 70)
    print("TEACHER V2 EVALUATION")
    print("=" * 70)

    print("Samples:", len(all_labels))
    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("Confusion matrix")
    print("----------------")
    print("Rows = true labels")
    print("Columns = predicted labels")
    print()
    print(cm)
    print()

    print("Predicted class distribution")
    print("----------------------------")

    for class_id in [0, 1, 2]:

        count = sum(
            prediction == class_id
            for prediction in all_predictions
        )

        print(
            f"Class {class_id}: {count}"
        )

    print()

    print("Classification report")
    print("---------------------")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            target_names=[
                "Normal",
                "Osteopenia",
                "Osteoporosis",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 70)
    print("TEACHER EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()