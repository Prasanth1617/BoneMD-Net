import torch
from collections import Counter
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


device = torch.device("cuda")

dataset = CachedMultimodalDataset(
    "cache/val",
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

model = BoneMDTeacherContrastive().to(device)
model.eval()

predictions = []
labels = []

with torch.no_grad():

    for batch in loader:

        ap = batch["ap"].to(device)
        lateral = batch["lateral"].to(device)
        ct = batch["ct"].to(device)

        output = model(
            ap,
            lateral,
            ct
        )

        pred = output["logits"].argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )

        labels.extend(
            batch["label"].tolist()
        )


print()
print("=== VALIDATION PREDICTION DISTRIBUTION ===")
print("Actual:    ", Counter(labels))
print("Predicted: ", Counter(predictions))

print()
print("Patient-level predictions:")

for i, (actual, predicted) in enumerate(
    zip(labels, predictions),
    start=1
):
    print(
        f"{i:02d}: actual={actual} predicted={predicted}"
    )