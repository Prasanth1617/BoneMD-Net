import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
LR = 1e-4

print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Learning rate:", LR)
print("Loss: ordinary CrossEntropyLoss")

train_dataset = CachedMultimodalDataset("cache/train")
val_dataset = CachedMultimodalDataset("cache/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
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
    weight_decay=1e-4,
)

for epoch in range(1, EPOCHS + 1):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for batch in train_loader:

        ap = batch["ap"].to(DEVICE)
        lateral = batch["lateral"].to(DEVICE)
        ct = batch["ct"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(ap, lateral, ct)

        loss = criterion(
            outputs["logits"],
            labels,
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs["logits"].argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    confusion = torch.zeros(
        3,
        3,
        dtype=torch.int64,
    )

    with torch.no_grad():

        for batch in val_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE).long()

            outputs = model(
                ap,
                lateral,
                ct,
            )

            loss = criterion(
                outputs["logits"],
                labels,
            )

            val_loss += loss.item()

            predictions = outputs["logits"].argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            confusion[
                labels.item(),
                predictions.item()
            ] += 1

    val_accuracy = val_correct / val_total
    val_loss /= len(val_loader)

    print()
    print(f"Epoch {epoch:02d}")
    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f}"
    )
    print(
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )

    print("Validation predictions:", confusion.sum(dim=0).tolist())

print()
print("Diagnostic training completed.")