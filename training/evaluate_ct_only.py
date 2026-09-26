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
from models.ct_only_student import CTOnlyStudent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/ct_only_best.pth"
TEST_CACHE = "cache/test"


def main():

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("Test cache:", TEST_CACHE)
    print()

    dataset = CachedMultimodalDataset(TEST_CACHE)

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

    model = CTOnlyStudent(
        feature_dim=256,
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"],
    )
    print()

    all_labels = []
    all_predictions = []
    all_patient_ids = []

    with torch.no_grad():

        for batch in loader:

            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = model(ct)

            logits = output["logits"]

            predictions = logits.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_patient_ids.extend(
                batch["patient_id"]
                if isinstance(
                    batch["patient_id"],
                    list,
                )
                else batch["patient_id"].cpu().tolist()
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
    print("CT-ONLY TEST EVALUATION")
    print("=" * 70)

    print("Checkpoint epoch:", checkpoint["epoch"])
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
            1
            for prediction in all_predictions
            if prediction == class_id
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
    print("CT-ONLY TEST EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
