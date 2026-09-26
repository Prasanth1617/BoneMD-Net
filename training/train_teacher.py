import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher
from training.config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_WORKERS,
)


RANDOM_SEED = 42
NUM_EPOCHS = 30

CHECKPOINT_PATH = "checkpoints/teacher_best_v2.pth"

EARLY_STOPPING_PATIENCE = 7

BEST_VAL_ACCURACY = -1.0


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for batch in loader:
        ap = batch["ap"].to(DEVICE, non_blocking=True)
        lateral = batch["lateral"].to(DEVICE, non_blocking=True)
        ct = batch["ct"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(ap, lateral, ct)

            loss = criterion(
                output["logits"],
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

        predictions = output["logits"].argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics):
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)


def main():
    set_seed()

    print("Device:", DEVICE)
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Checkpoint:", CHECKPOINT_PATH)

    train_dataset = CachedMultimodalDataset("cache/train")
    val_dataset = CachedMultimodalDataset("cache/val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Training distribution:
    # class 0 = 85
    # class 1 = 55
    # class 2 = 50
    #
    # Inverse-frequency weights, normalized around 1.0.
    class_counts = torch.tensor(
        [85.0, 55.0, 50.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    class_weights = class_counts.sum() / (
        NUM_CLASSES * class_counts
    )

    print(
        "Class weights:",
        [round(float(x), 4) for x in class_weights],
    )

    model = BoneMDTeacher(
        feature_dim=512,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_val_accuracy = BEST_VAL_ACCURACY
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        print()
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
            )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.4f}"
        )

        print(f"Learning rate: {current_lr:.7f}")

        current_metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
        }

        improved = False

        if val_metrics["accuracy"] > best_val_accuracy:
            improved = True
        elif (
            val_metrics["accuracy"] == best_val_accuracy
            and val_metrics["loss"] < best_val_loss
        ):
            improved = True

        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                current_metrics,
            )

            print(
                "New best validation accuracy:",
                round(best_val_accuracy, 4),
            )

            print(
                "Best checkpoint saved:",
                CHECKPOINT_PATH,
            )

        else:
            epochs_without_improvement += 1

            print(
                "Best validation accuracy remains:",
                round(best_val_accuracy, 4),
            )

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
                "/",
                EARLY_STOPPING_PATIENCE,
            )

        if DEVICE.type == "cuda":
            peak_memory = (
                torch.cuda.max_memory_allocated(DEVICE)
                / 1024**2
            )

            print(
                "Peak VRAM:",
                round(peak_memory, 1),
                "MiB",
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print()
            print(
                "Early stopping triggered after",
                epoch,
                "epochs.",
            )
            break

    print()
    print("Teacher V2 training completed successfully")
    print("Best validation accuracy:", round(best_val_accuracy, 4))
    print("Best checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
