import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.lateral_only_student import LateralOnlyStudent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CACHE = "cache/train"
VAL_CACHE = "cache/val"

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30

CHECKPOINT = "checkpoints/lateral_only_best.pth"

train_dataset = CachedMultimodalDataset(TRAIN_CACHE)
val_dataset = CachedMultimodalDataset(VAL_CACHE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = LateralOnlyStudent().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

best_val_acc = -1.0
best_epoch = 0

print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print("Model: Lateral-only student")
print(f"Checkpoint: {CHECKPOINT}")
print()
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print()

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        lateral = batch["lateral"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(lateral)
        logits = outputs["logits"]

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            lateral = batch["lateral"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(lateral)
            logits = outputs["logits"]

            loss = criterion(logits, labels)

            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
            },
            CHECKPOINT,
        )

        print(f"    New best Lateral-only checkpoint saved (epoch {epoch})")

print()
print("=" * 70)
print("LATERAL-ONLY TRAINING COMPLETED")
print("=" * 70)
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc}")
print(f"Checkpoint: {CHECKPOINT}")
