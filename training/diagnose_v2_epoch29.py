import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 29
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("V2 EPOCH-29 CONFUSION MATRIX DIAGNOSTIC")
    print("=" * 70)

    train_dataset = CachedMultimodalDataset(
        "cache/train"
    )

    val_dataset = CachedMultimodalDataset(
        "cache/val"
    )

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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("Reproducing training through epoch 29...")
    print()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

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

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} completed"
            )

    print()
    print("=" * 70)
    print("VALIDATION DIAGNOSTIC")
    print("=" * 70)

    model.eval()

    confusion = torch.zeros(
        3,
        3,
        dtype=torch.long,
    )

    predictions = []
    labels_list = []

    with torch.no_grad():

        for batch in val_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            label = int(batch["label"].item())

            output = model(
                ap,
                lateral,
                ct,
            )

            prediction = int(
                output["logits"]
                .argmax(dim=1)
                .item()
            )

            confusion[
                label,
                prediction
            ] += 1

            predictions.append(prediction)
            labels_list.append(label)

    accuracy = sum(
        p == y
        for p, y in zip(
            predictions,
            labels_list,
        )
    ) / len(labels_list)

    print()
    print("Validation accuracy:")
    print(f"{accuracy:.4f}")

    print()
    print("Confusion matrix:")
    print(confusion)

    print()
    print("True class counts:")

    for class_id in range(3):
        print(
            f"Class {class_id}: "
            f"{labels_list.count(class_id)}"
        )

    print()
    print("Predicted class counts:")

    for class_id in range(3):
        print(
            f"Class {class_id}: "
            f"{predictions.count(class_id)}"
        )

    print()
    print("Per-class recall:")

    for class_id in range(3):

        total = confusion[class_id].sum().item()

        if total > 0:
            recall = (
                confusion[class_id, class_id].item()
                / total
            )
        else:
            recall = 0.0

        print(
            f"Class {class_id}: "
            f"{recall:.4f}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()