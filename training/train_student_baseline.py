import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_student import BoneMDStudent


SEED = 42
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "checkpoints/student_baseline_best.pth"


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate(student, loader, criterion):
    student.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = student(ap, lateral, ct)

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
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Loss: CrossEntropyLoss")
    print("Checkpoint:", CHECKPOINT_PATH)
    print()

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    train_dataset = CachedMultimodalDataset("cache/train")
    val_dataset = CachedMultimodalDataset("cache/val")

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

    # ------------------------------------------------------------
    # Student
    # ------------------------------------------------------------
    student = BoneMDStudent(
        feature_dim=256,
        num_classes=3,
    ).to(DEVICE)

    print(
        "Student parameters:",
        sum(p.numel() for p in student.parameters()),
    )
    print()

    # ------------------------------------------------------------
    # Standard supervised classification loss
    # ------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0
    best_epoch = 0

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):

        student.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            output = student(
                ap,
                lateral,
                ct,
            )

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
            student,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        # --------------------------------------------------------
        # Save best baseline
        # --------------------------------------------------------
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"    New best baseline checkpoint saved "
                f"(epoch {epoch})"
            )

    print()
    print("=" * 70)
    print("STUDENT BASELINE TRAINING COMPLETED")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", best_val_accuracy)
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
