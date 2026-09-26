import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_norm import BoneMDTeacherNorm


SEED = 42
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = model(ap, lateral, ct)

            loss = criterion(
                output["logits"],
                labels,
            )

            running_loss += loss.item()

            predictions = output["logits"].argmax(dim=1)

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

    return running_loss / total, correct / total


def main():

    print("Device:", DEVICE)
    print("Experiment: Normalized Teacher + Class-Weighted Loss")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print()

    train_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "train"
    )

    val_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print()

    model = BoneMDTeacherNorm(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    # Exact inverse-frequency weights calculated
    # from the 190-patient training split.
    class_weights = torch.tensor(
        [
            0.7450980392156863,
            1.1515151515151516,
            1.2666666666666666,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Class weights:", class_weights.tolist())
    print()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            output = model(ap, lateral, ct)

            loss = criterion(
                output["logits"],
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            predictions = output["logits"].argmax(dim=1)

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_weights": class_weights.detach().cpu(),
                },
                PROJECT_ROOT
                / "checkpoints"
                / "teacher_norm_weighted_best.pth",
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f}"
        )

    print()
    print("=" * 70)
    print("WEIGHTED NORMALIZED TEACHER EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best epoch: {best_epoch}")
    print()
    print("Checkpoint:")
    print(
        PROJECT_ROOT
        / "checkpoints"
        / "teacher_norm_weighted_best.pth"
    )


if __name__ == "__main__":
    main()
