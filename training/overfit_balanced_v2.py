import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 100
LEARNING_RATE = 1e-4

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("BALANCED 12-PATIENT V2 GROUPNORM OVERFIT TEST")
    print("=" * 70)

    dataset = CachedMultimodalDataset("cache/train")

    class_indices = {
        0: [],
        1: [],
        2: [],
    }

    for index in range(len(dataset)):

        label = int(dataset[index]["label"])

        if len(class_indices[label]) < 4:
            class_indices[label].append(index)

        if all(
            len(values) == 4
            for values in class_indices.values()
        ):
            break

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

    print("Selected samples:", len(subset))
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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    print()
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay: 0")
    print()

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

            running_loss += (
                loss.detach().item()
            )

            predictions = output[
                "logits"
            ].argmax(dim=1)

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

        accuracy = correct / total
        average_loss = running_loss / total

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {average_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

    print()
    print("=" * 70)
    print("FINAL TRAIN MODE EVALUATION")
    print("=" * 70)

    model.train()

    train_predictions = []
    labels_list = []

    with torch.no_grad():

        for batch in evaluation_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

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

            train_predictions.append(
                prediction
            )

            labels_list.append(
                int(batch["label"].item())
            )

    train_accuracy = sum(
        p == y
        for p, y in zip(
            train_predictions,
            labels_list,
        )
    ) / len(labels_list)

    print(
        "Train-mode accuracy:",
        f"{train_accuracy:.4f}",
    )

    print(
        "Predictions:",
        train_predictions,
    )

    print(
        "Labels:     ",
        labels_list,
    )

    print()
    print("=" * 70)
    print("FINAL EVAL MODE EVALUATION")
    print("=" * 70)

    model.eval()

    eval_predictions = []

    with torch.no_grad():

        for batch in evaluation_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

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

            eval_predictions.append(
                prediction
            )

    eval_accuracy = sum(
        p == y
        for p, y in zip(
            eval_predictions,
            labels_list,
        )
    ) / len(labels_list)

    print(
        "Eval-mode accuracy:",
        f"{eval_accuracy:.4f}",
    )

    print(
        "Predictions:",
        eval_predictions,
    )

    print(
        "Labels:     ",
        labels_list,
    )

    print()
    print("=" * 70)
    print("V2 GROUPNORM OVERFIT TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()