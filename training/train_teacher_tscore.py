import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_tscore import BoneMDTeacherTScore


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_CACHE = PROJECT_ROOT / "cache" / "train"
VAL_CACHE = PROJECT_ROOT / "cache" / "val"

CLINICAL_FILE = PROJECT_ROOT / "dataset" / "lumos_clinical_data.xlsx"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "teacher_tscore_best.pth"

EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Auxiliary T-score loss weight.
LAMBDA_TSCORE = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TSCORE_COLUMNS = [
    "T_L1-L2",
    "T_L1-L3",
    "T_L1-L4",
    "T_L2-L3",
    "T_L2-L4",
    "T_L3-L4",
]

# Statistics calculated ONLY from the training split.
TSCORE_MEAN = np.array([
    -1.152381,
    -1.116931,
    -0.991534,
    -1.145503,
    -0.964550,
    -0.791005,
], dtype=np.float32)

TSCORE_STD = np.array([
    1.768235,
    1.865142,
    1.959058,
    1.998708,
    2.089961,
    2.221408,
], dtype=np.float32)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class TeacherTScoreDataset(Dataset):

    def __init__(self, cache_dir, clinical_df):
        self.cache_dir = Path(cache_dir)

        self.files = sorted(
            self.cache_dir.glob("patient_*.pt")
        )

        if not self.files:
            raise RuntimeError(
                f"No cached patient files found in {self.cache_dir}"
            )

        self.clinical = clinical_df.copy()

        self.clinical["patient_id"] = (
            self.clinical["patient_id"].astype(int)
        )

        self.clinical_lookup = (
            self.clinical
            .set_index("patient_id")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]

        sample = torch.load(
            path,
            map_location="cpu",
        )

        patient_id = int(sample["patient_id"])

        row = self.clinical_lookup.loc[patient_id]

        tscore_values = row[TSCORE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        valid_tscore = np.isfinite(tscore_values).all()

        if valid_tscore:
            normalized_tscore = (
                (tscore_values - TSCORE_MEAN)
                / TSCORE_STD
            )

            tscore_tensor = torch.tensor(
                normalized_tscore,
                dtype=torch.float32,
            )

            tscore_mask = torch.tensor(
                True,
                dtype=torch.bool,
            )

        else:
            # Patient 169 has missing T-scores.
            # Keep the patient for classification,
            # but exclude it from auxiliary regression loss.
            tscore_tensor = torch.zeros(
                6,
                dtype=torch.float32,
            )

            tscore_mask = torch.tensor(
                False,
                dtype=torch.bool,
            )

        return {
            "ap": sample["ap"].float(),
            "lateral": sample["lateral"].float(),
            "ct": sample["ct"].float(),
            "label": sample["label"].long(),
            "patient_id": patient_id,
            "tscores": tscore_tensor,
            "tscore_mask": tscore_mask,
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, criterion_cls, criterion_tscore):

    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_tscore_loss = 0.0

    correct = 0
    total = 0

    valid_tscore_samples = 0

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            tscores = batch["tscores"].to(DEVICE)
            tscore_mask = batch["tscore_mask"].to(DEVICE)

            output = model(
                ap,
                lateral,
                ct,
            )

            logits = output["logits"]
            predicted = torch.argmax(
                logits,
                dim=1,
            )

            cls_loss = criterion_cls(
                logits,
                labels,
            )

            if tscore_mask.any():

                tscore_loss = criterion_tscore(
                    output["tscore_predictions"][tscore_mask],
                    tscores[tscore_mask],
                )

                valid_tscore_samples += int(
                    tscore_mask.sum().item()
                )

            else:

                tscore_loss = torch.tensor(
                    0.0,
                    device=DEVICE,
                )

            loss = (
                cls_loss
                + LAMBDA_TSCORE * tscore_loss
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_cls_loss += (
                cls_loss.item() * batch_size
            )

            total_tscore_loss += (
                tscore_loss.item() * batch_size
            )

            correct += (
                (predicted == labels)
                .sum()
                .item()
            )

            total += batch_size

    return {
        "loss": total_loss / total,
        "classification_loss": total_cls_loss / total,
        "tscore_loss": total_tscore_loss / total,
        "accuracy": correct / total,
        "valid_tscore_samples": valid_tscore_samples,
    }


# ============================================================
# Training
# ============================================================

def main():

    set_seed(SEED)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load clinical data
    # --------------------------------------------------------

    clinical_df = pd.read_excel(
        CLINICAL_FILE
    )

    print(
        "Clinical rows:",
        len(clinical_df),
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = TeacherTScoreDataset(
        TRAIN_CACHE,
        clinical_df,
    )

    val_dataset = TeacherTScoreDataset(
        VAL_CACHE,
        clinical_df,
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # Verify missing T-score handling.
    missing_train = 0

    for i in range(len(train_dataset)):

        sample = train_dataset[i]

        if not bool(sample["tscore_mask"].item()):
            missing_train += 1

    print(
        "Training samples without T-scores:",
        missing_train,
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BoneMDTeacherTScore(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------

    classification_criterion = nn.CrossEntropyLoss()

    tscore_criterion = nn.SmoothL1Loss(
        beta=1.0,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_epoch = -1

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_tscore_loss = 0.0

        correct = 0
        total = 0

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            tscores = batch["tscores"].to(DEVICE)
            tscore_mask = batch["tscore_mask"].to(DEVICE)

            optimizer.zero_grad()

            output = model(
                ap,
                lateral,
                ct,
            )

            logits = output["logits"]

            # Primary classification objective.
            cls_loss = classification_criterion(
                logits,
                labels,
            )

            # Auxiliary T-score objective.
            if tscore_mask.any():

                tscore_loss = tscore_criterion(
                    output["tscore_predictions"][tscore_mask],
                    tscores[tscore_mask],
                )

            else:

                tscore_loss = torch.tensor(
                    0.0,
                    device=DEVICE,
                )

            # Combined objective.
            loss = (
                cls_loss
                + LAMBDA_TSCORE * tscore_loss
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            running_cls_loss += (
                cls_loss.item() * batch_size
            )

            running_tscore_loss += (
                tscore_loss.item() * batch_size
            )

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                (predicted == labels)
                .sum()
                .item()
            )

            total += batch_size

        train_loss = running_loss / total
        train_cls_loss = running_cls_loss / total
        train_tscore_loss = running_tscore_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            classification_criterion,
            tscore_criterion,
        )

        print(
            f"Epoch {epoch:02d} "
            f"TrainLoss {train_loss:.4f} "
            f"TrainCls {train_cls_loss:.4f} "
            f"TrainTS {train_tscore_loss:.4f} "
            f"TrainAcc {train_accuracy:.4f} "
            f"ValLoss {val_metrics['loss']:.4f} "
            f"ValCls {val_metrics['classification_loss']:.4f} "
            f"ValTS {val_metrics['tscore_loss']:.4f} "
            f"ValAcc {val_metrics['accuracy']:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint using validation accuracy only.
        # ----------------------------------------------------

        if val_metrics["accuracy"] > best_val_accuracy:

            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    "train_loss": train_loss,
                    "train_classification_loss": train_cls_loss,
                    "train_tscore_loss": train_tscore_loss,
                    "train_accuracy": train_accuracy,

                    "val_loss": val_metrics["loss"],
                    "val_classification_loss": (
                        val_metrics["classification_loss"]
                    ),
                    "val_tscore_loss": (
                        val_metrics["tscore_loss"]
                    ),
                    "val_accuracy": val_metrics["accuracy"],

                    "lambda_tscore": LAMBDA_TSCORE,

                    "tscore_columns": TSCORE_COLUMNS,
                    "tscore_mean": TSCORE_MEAN.tolist(),
                    "tscore_std": TSCORE_STD.tolist(),

                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  -> Saved best checkpoint "
                f"(epoch {epoch}, "
                f"val acc {val_metrics['accuracy']:.4f})"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("Training complete.")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()