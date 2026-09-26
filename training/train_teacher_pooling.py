import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.bone_md_teacher_pooling import BoneMDTeacherPooling
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CACHE = PROJECT_ROOT / "cache" / "train"
VAL_CACHE = PROJECT_ROOT / "cache" / "val"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_pooling_best.pth"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

EPOCHS = 50
BATCH_SIZE = 2

LR = 1e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.10

QUEUE_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Queue-based supervised contrastive loss
# ============================================================

def queue_supervised_contrastive_loss(
    features,
    labels,
    queue_features,
    queue_labels,
    temperature=0.07,
):
    """
    Cross-batch supervised contrastive loss.

    Current batch features are contrasted against:
    - other samples in the current batch
    - stored samples in the feature queue

    Same-class samples are positives.
    Different-class samples are negatives.
    """

    device = features.device

    batch_size = features.size(0)

    if queue_features is None or queue_features.size(0) == 0:
        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True,
        )

    all_features = torch.cat(
        [features, queue_features],
        dim=0,
    )

    all_labels = torch.cat(
        [labels, queue_labels],
        dim=0,
    )

    # Similarity of current batch against
    # current batch + queue.
    logits = torch.matmul(
        features,
        all_features.T,
    ) / temperature

    # Prevent each sample from being its own positive.
    self_mask = torch.zeros(
        (batch_size, all_features.size(0)),
        dtype=torch.bool,
        device=device,
    )

    self_mask[
        torch.arange(batch_size, device=device),
        torch.arange(batch_size, device=device),
    ] = True

    positive_mask = (
        labels.unsqueeze(1)
        == all_labels.unsqueeze(0)
    )

    positive_mask = positive_mask & (~self_mask)

    valid_rows = positive_mask.any(dim=1)

    if not valid_rows.any():
        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True,
        )

    logits = logits.masked_fill(
        self_mask,
        -1e9,
    )

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    positive_log_prob = (
        log_prob * positive_mask.float()
    ).sum(dim=1)

    positive_count = positive_mask.sum(dim=1)

    loss = (
        -positive_log_prob[valid_rows]
        / positive_count[valid_rows].float()
    ).mean()

    return loss


# ============================================================
# Data loading
# ============================================================

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
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# Model
# ============================================================

model = BoneMDTeacherPooling().to(DEVICE)


# ============================================================
# Class weights
# ============================================================

class_counts = torch.zeros(
    3,
    dtype=torch.float32,
)

for i in range(len(train_dataset)):
    sample = train_dataset[i]

    label = int(
        sample["label"]
    )

    class_counts[label] += 1


class_weights = (
    class_counts.sum()
    / (
        3.0 * class_counts
    )
)

class_weights = class_weights.to(
    DEVICE
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Feature queue
# ============================================================

queue_features = None
queue_labels = None


def update_queue(
    features,
    labels,
):
    global queue_features
    global queue_labels

    features = features.detach()
    labels = labels.detach()

    if queue_features is None:
        queue_features = features.clone()
        queue_labels = labels.clone()
    else:
        queue_features = torch.cat(
            [
                queue_features,
                features,
            ],
            dim=0,
        )

        queue_labels = torch.cat(
            [
                queue_labels,
                labels,
            ],
            dim=0,
        )

    if queue_features.size(0) > QUEUE_SIZE:
        queue_features = queue_features[
            -QUEUE_SIZE:
        ]

        queue_labels = queue_labels[
            -QUEUE_SIZE:
        ]


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

print("=" * 70)
print("BoneMD-Net Teacher — Average + Max X-ray Pooling")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Temperature: {TEMPERATURE}")
print(
    f"Contrastive weight: "
    f"{CONTRASTIVE_WEIGHT}"
)
print(f"Queue size: {QUEUE_SIZE}")
print(
    f"Class counts: "
    f"{class_counts.tolist()}"
)
print(
    f"Class weights: "
    f"{class_weights.tolist()}"
)

print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters())}"
)

print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    train_loss = 0.0
    train_ce_loss = 0.0
    train_contrastive_loss = 0.0

    train_correct = 0
    train_total = 0

    for batch in train_loader:

        ap = batch["ap"].to(
            DEVICE,
            non_blocking=True,
        )

        lateral = batch["lateral"].to(
            DEVICE,
            non_blocking=True,
        )

        ct = batch["ct"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            ap,
            lateral,
            ct,
        )

        logits = outputs["logits"]

        contrastive_features = (
            outputs["contrastive_features"]
        )

        ce_loss = criterion(
            logits,
            labels,
        )

        con_loss = (
            queue_supervised_contrastive_loss(
                contrastive_features,
                labels,
                queue_features,
                queue_labels,
                TEMPERATURE,
            )
        )

        loss = (
            ce_loss
            + CONTRASTIVE_WEIGHT * con_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        update_queue(
            contrastive_features,
            labels,
        )

        predictions = logits.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        train_loss += (
            loss.item()
            * labels.size(0)
        )

        train_ce_loss += (
            ce_loss.item()
            * labels.size(0)
        )

        train_contrastive_loss += (
            con_loss.item()
            * labels.size(0)
        )

    train_loss /= train_total
    train_ce_loss /= train_total
    train_contrastive_loss /= train_total

    train_accuracy = (
        train_correct
        / train_total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            ap = batch["ap"].to(
                DEVICE,
                non_blocking=True,
            )

            lateral = batch["lateral"].to(
                DEVICE,
                non_blocking=True,
            )

            ct = batch["ct"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(
                ap,
                lateral,
                ct,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                labels,
            )

            predictions = logits.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            val_loss += (
                loss.item()
                * labels.size(0)
            )

    val_loss /= val_total

    val_accuracy = (
        val_correct
        / val_total
    )


    # ========================================================
    # Logging
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"CE {train_ce_loss:.4f} | "
        f"SupCon {train_contrastive_loss:.4f} | "
        f"Train Acc {train_accuracy:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_accuracy:.4f}"
    )


    # ========================================================
    # Save best checkpoint
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "temperature": TEMPERATURE,
            "contrastive_weight": CONTRASTIVE_WEIGHT,
            "queue_size": QUEUE_SIZE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

        print(
            f"  Saved best checkpoint -> "
            f"{CHECKPOINT_PATH}"
        )


print("=" * 70)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)
print("=" * 70)