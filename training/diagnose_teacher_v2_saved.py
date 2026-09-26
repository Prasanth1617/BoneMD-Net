import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "checkpoints/teacher_v2_controlled_best.pth"


def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print()

    # ------------------------------------------------------------
    # Load validation dataset
    # ------------------------------------------------------------
    val_dataset = CachedMultimodalDataset("cache/val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Validation samples:", len(val_dataset))
    print()

    # ------------------------------------------------------------
    # Build exact V2 teacher architecture
    # ------------------------------------------------------------
    model = BoneMDTeacher(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    # ------------------------------------------------------------
    # Load saved checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Saved train loss:", checkpoint["train_loss"])
    print("Saved train accuracy:", checkpoint["train_accuracy"])
    print("Saved val loss:", checkpoint["val_loss"])
    print("Saved val accuracy:", checkpoint["val_accuracy"])
    print()

    # ------------------------------------------------------------
    # Validation predictions
    # ------------------------------------------------------------
    all_labels = []
    all_predictions = []
    all_patient_ids = []

    with torch.no_grad():
        for batch in val_loader:
            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = model(ap, lateral, ct)
            logits = output["logits"]

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_patient_ids.extend(batch["patient_id"])

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
    )

    print("=" * 70)
    print("VALIDATION CONFUSION MATRIX")
    print("=" * 70)
    print(cm)
    print()

    # ------------------------------------------------------------
    # Prediction distribution
    # ------------------------------------------------------------
    prediction_counts = {
        0: all_predictions.count(0),
        1: all_predictions.count(1),
        2: all_predictions.count(2),
    }

    true_counts = {
        0: all_labels.count(0),
        1: all_labels.count(1),
        2: all_labels.count(2),
    }

    print("True label counts:")
    print(true_counts)
    print()

    print("Prediction counts:")
    print(prediction_counts)
    print()

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------
    print("=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    report = classification_report(
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

    print(report)

    # ------------------------------------------------------------
    # Patient-level prediction listing
    # ------------------------------------------------------------
    print("=" * 70)
    print("PATIENT-LEVEL VALIDATION PREDICTIONS")
    print("=" * 70)

    for patient_id, true_label, prediction in zip(
        all_patient_ids,
        all_labels,
        all_predictions,
    ):
        status = "CORRECT" if true_label == prediction else "WRONG"

        print(
            f"Patient {patient_id:>3} | "
            f"True: {true_label} | "
            f"Pred: {prediction} | "
            f"{status}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()