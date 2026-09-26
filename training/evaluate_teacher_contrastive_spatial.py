import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive_spatial import (
    BoneMDTeacherContrastiveSpatial,
)
from dataset.cached_multimodal_dataset import (
    CachedMultimodalDataset,
)


TEST_CACHE = PROJECT_ROOT / "cache" / "test"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_spatial_100_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BoneMD-Net Spatial Contrastive Teacher - LOCKED TEST")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

dataset = CachedMultimodalDataset(TEST_CACHE)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

model = BoneMDTeacherContrastiveSpatial(
    feature_dim=512,
    num_classes=3,
    contrastive_dim=128,
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch in loader:

        ap = batch["ap"].to(DEVICE)
        lateral = batch["lateral"].to(DEVICE)
        ct = batch["ct"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(
            ap,
            lateral,
            ct,
        )

        logits = outputs["logits"]

        predictions = logits.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2],
)

print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test samples: {len(all_labels)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print()
print("Actual distribution:")
print(Counter(all_labels))

print()
print("Prediction distribution:")
print(Counter(all_predictions))

print()
print("Confusion Matrix")
print("(rows = actual, columns = predicted)")
print(cm)

print()
print("Classification Report")
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
        zero_division=0,
    )
)

print("=" * 70)