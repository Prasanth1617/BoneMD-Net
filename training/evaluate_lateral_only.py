import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.lateral_only_student import LateralOnlyStudent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CACHE = "cache/test"
CHECKPOINT = "checkpoints/lateral_only_best.pth"

dataset = CachedMultimodalDataset(TEST_CACHE)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = LateralOnlyStudent().to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in loader:
        lateral = batch["lateral"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(lateral)
        predictions = torch.argmax(outputs["logits"], dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

accuracy = sum(
    p == y for p, y in zip(all_predictions, all_labels)
) / len(all_labels)

print("=" * 70)
print("LATERAL-ONLY TEST EVALUATION")
print("=" * 70)
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Test samples: {len(all_labels)}")
print(f"Accuracy: {accuracy:.4f} = {accuracy * 100:.2f}%")
print()
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_predictions))
print()
print("Classification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Normal", "Osteopenia", "Osteoporosis"],
        digits=4,
        zero_division=0,
    )
)
print()
print("Predicted class distribution:")
for cls in range(3):
    print(f"Class {cls}: {all_predictions.count(cls)}")
print("=" * 70)
