import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/teacher_best_v2.pth"


print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)

dataset = CachedMultimodalDataset("cache/val")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

model = BoneMDTeacher(
    feature_dim=512,
    num_classes=3,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

confusion = torch.zeros(3, 3, dtype=torch.int64)

with torch.no_grad():

    for batch in loader:

        ap = batch["ap"].to(DEVICE)
        lateral = batch["lateral"].to(DEVICE)
        ct = batch["ct"].to(DEVICE)

        label = batch["label"].to(DEVICE).long()

        output = model(
            ap,
            lateral,
            ct,
        )

        logits = output["logits"]

        prediction = logits.argmax(dim=1)

        confusion[label.item(), prediction.item()] += 1


print()
print("Confusion Matrix")
print("Rows = Actual")
print("Columns = Predicted")
print()
print(confusion)

print()

for cls in range(3):
    actual = confusion[cls].sum().item()
    correct = confusion[cls, cls].item()

    if actual > 0:
        accuracy = correct / actual
    else:
        accuracy = 0.0

    print(
        f"Class {cls}: "
        f"actual={actual}, "
        f"correct={correct}, "
        f"class_accuracy={accuracy:.4f}"
    )

print()

predicted_counts = confusion.sum(dim=0)

print("Predicted class distribution:")

for cls in range(3):
    print(
        f"Class {cls}: "
        f"{predicted_counts[cls].item()}"
    )

print()

actual_counts = confusion.sum(dim=1)

print("Actual class distribution:")

for cls in range(3):
    print(
        f"Class {cls}: "
        f"{actual_counts[cls].item()}"
    )