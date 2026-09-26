import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.ct_only_student import CTOnlyStudent


RANDOM_SEED = 42

TRAIN_CACHE = "cache/train"
VAL_CACHE = "cache/val"

BATCH_SIZE = 1
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30

CHECKPOINT = "checkpoints/ct_only_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    torch.manual_seed(RANDOM_SEED)

    print("Device:", DEVICE)
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Model: CT-only student")
    print("Checkpoint:", CHECKPOINT)
    print()

    train_dataset = CachedMultimodalDataset(
        TRAIN_CACHE
    )

    val_dataset = CachedMultimodalDataset(
        VAL_CACHE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print()

    model = CTOnlyStudent(
        feature_dim=256,
        num_classes=3,
    ).to(DEVICE)

    print(
        "Model parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )
    print()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:

            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            output = model(ct)

            logits = output["logits"]

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = (
            train_correct / train_total
        )

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for batch in val_loader:

                ct = batch["ct"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                output = model(ct)

                logits = output["logits"]

                loss = criterion(
                    logits,
                    labels,
                )

                val_loss += (
                    loss.item()
                    * labels.size(0)
                )

                predictions = logits.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = (
            val_correct / val_total
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                CHECKPOINT,
            )

            print(
                f"    New best CT-only checkpoint saved "
                f"(epoch {epoch})"
            )

    print()
    print("=" * 70)
    print("CT-ONLY TRAINING COMPLETED")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print(
        "Best validation accuracy:",
        best_val_accuracy,
    )
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
