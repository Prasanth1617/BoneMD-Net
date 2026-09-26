import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_norm import BoneMDTeacherNorm


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 250
BATCH_SIZE = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# IMPORTANT:
# This is a NEW checkpoint.
# The existing 75.61% model remains untouched.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_norm_osteopenia_smooth_cosine_best.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# EVALUATION
# ============================================================

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

            output = model(
                ap,
                lateral,
                ct,
            )

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

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BONE-MD-NET")
    print("FINAL ACCURACY IMPROVEMENT EXPERIMENT")
    print("=" * 70)

    print()
    print("Device:", DEVICE)
    print("Experiment:")
    print("  Normalized Teacher")
    print("  Strong Osteopenia Weight")
    print("  Label Smoothing")
    print("  Cosine Learning Rate")
    print()
    print("Seed:", SEED)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Label smoothing:", LABEL_SMOOTHING)
    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = BoneMDTeacherNorm(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    # --------------------------------------------------------
    # CLASS WEIGHTS
    #
    # 0 = Normal
    # 1 = Osteopenia
    # 2 = Osteoporosis
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            0.70,   # Normal
            1.80,   # Osteopenia
            1.20,   # Osteoporosis
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Class weights:")
    print("  Normal       :", class_weights[0].item())
    print("  Osteopenia   :", class_weights[1].item())
    print("  Osteoporosis :", class_weights[2].item())
    print()

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # COSINE LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_epoch = -1

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

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

            output = model(
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

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        # ----------------------------------------------------
        # UPDATE LEARNING RATE
        # ----------------------------------------------------

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_weights": class_weights.detach().cpu(),
                    "label_smoothing": LABEL_SMOOTHING,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "experiment": (
                        "Normalized Teacher + "
                        "Strong Osteopenia Weight + "
                        "Label Smoothing + "
                        "Cosine LR"
                    ),
                },
                CHECKPOINT_PATH,
            )

            best_marker = "  <-- BEST"

        else:

            best_marker = ""

        # ----------------------------------------------------
        # PRINT PROGRESS
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"LR: {current_lr:.7f} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f}"
            f"{best_marker}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best validation accuracy (%): "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print()
    print("Experiment configuration:")
    print("  Normal weight       = 0.70")
    print("  Osteopenia weight   = 1.80")
    print("  Osteoporosis weight = 1.20")
    print("  Label smoothing     =", LABEL_SMOOTHING)
    print("  Initial LR          =", LEARNING_RATE)
    print("  Minimum LR          = 1e-6")
    print("  Weight decay        =", WEIGHT_DECAY)
    print("  Scheduler           = CosineAnnealingLR")

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()