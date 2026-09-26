import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 23
LEARNING_RATE = 1e-4

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate(model, loader, mode_name):

    if mode_name == "train":
        model.train()
    else:
        model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(
                ap,
                lateral,
                ct,
            )

            pred = output["logits"].argmax(
                dim=1
            )

            predictions.append(
                int(pred.item())
            )

            labels.append(
                int(batch["label"].item())
            )

    confusion = torch.zeros(
        3,
        3,
        dtype=torch.long,
    )

    for true, pred in zip(labels, predictions):
        confusion[true, pred] += 1

    accuracy = sum(
        p == y
        for p, y in zip(predictions, labels)
    ) / len(labels)

    print()
    print("=" * 70)
    print(f"{mode_name.upper()} MODE RESULTS")
    print("=" * 70)

    print("Accuracy:", f"{accuracy:.4f}")
    print("Predictions:", predictions)
    print("Labels:     ", labels)

    print()
    print("Confusion matrix:")
    print(confusion)

    return predictions


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("V2 GROUPNORM TRAIN/EVAL DIAGNOSTIC")
    print("=" * 70)

    dataset = CachedMultimodalDataset(
        "cache/train"
    )

    class_indices = {
        0: [],
        1: [],
        2: [],
    }

    for index in range(len(dataset)):

        label = int(dataset[index]["label"])

        if len(class_indices[label]) < 4:
            class_indices[label].append(index)

    selected_indices = (
        class_indices[0]
        + class_indices[1]
        + class_indices[2]
    )

    subset = Subset(
        dataset,
        selected_indices,
    )

    train_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    evaluation_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Selected samples:",
        len(subset),
    )

    print(
        "Class distribution:",
        {
            0: len(class_indices[0]),
            1: len(class_indices[1]),
            2: len(class_indices[2]),
        },
    )

    model = BoneMDTeacher(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    print()
    print(
        f"Training for exactly {EPOCHS} epochs..."
    )

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

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    evaluate(
        model,
        evaluation_loader,
        "train",
    )

    evaluate(
        model,
        evaluation_loader,
        "eval",
    )

    print()
    print("=" * 70)
    print("GROUPNORM DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()