import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 50
LR = 1e-4


print("Device:", DEVICE)

dataset = CachedMultimodalDataset("cache/train")

# Use first 10 training patients only
subset = Subset(dataset, list(range(10)))

loader = DataLoader(
    subset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

model = BoneMDTeacher(
    feature_dim=512,
    num_classes=3,
).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)

print("Diagnostic samples:", len(subset))
print("Epochs:", EPOCHS)
print("Learning rate:", LR)

for epoch in range(1, EPOCHS + 1):

    model.train()

    correct = 0
    total = 0
    running_loss = 0.0

    for batch in loader:

        ap = batch["ap"].to(DEVICE)
        lateral = batch["lateral"].to(DEVICE)
        ct = batch["ct"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            ap,
            lateral,
            ct,
        )

        loss = criterion(
            outputs["logits"],
            labels,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs["logits"].argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    if accuracy >= 0.99:
        print()
        print("SUCCESS: Teacher can overfit the 10-patient subset.")
        print("Stopping diagnostic.")
        break

print()
print("Diagnostic completed.")