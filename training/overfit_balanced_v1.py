import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 60
LEARNING_RATE = 1e-4

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("BALANCED 12-PATIENT V1 OVERFIT TEST")
    print("=" * 70)

    dataset = CachedMultimodalDataset("cache/train")

    class_indices = {
        0: [],
        1: [],
        2: [],
    }

    for index in range(len(dataset)):
        label = int(dataset[index]["label"])
        class_indices[label].append(index)

    selected_indices = []

    for class_id in range(3):

        indices = class_indices[class_id][:4]

        if len(indices) < 4:
            raise RuntimeError(
                f"Not enough samples for class {class_id}"
            )

        selected_indices.extend(indices)

    subset = Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    print("Selected samples:", len(subset))
    print(
        "Class distribution:",
        {
            0: len(class_indices[0][:4]),
            1: len(class_indices[1][:4]),
            2: len(class_indices[2][:4]),
        },
    )

    model = BoneMDTeacherV1(
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

        for batch in loader:

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

            logits = output["logits"]

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.detach().item()
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

        accuracy = correct / total
        average_loss = running_loss / total

        if (
            epoch == 1
            or epoch % 5 == 0
            or accuracy == 1.0
        ):
            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {average_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

        if accuracy == 1.0:
            print()
            print(
                f"100% training accuracy reached at "
                f"epoch {epoch}."
            )
            break

    print()
    print("=" * 70)
    print("FINAL PER-CLASS TRAINING RESULTS")
    print("=" * 70)

    model.eval()

    confusion = torch.zeros(
        3,
        3,
        dtype=torch.long,
    )

    with torch.no_grad():

        for batch in DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        ):

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            output = model(
                ap,
                lateral,
                ct,
            )

            predictions = output[
                "logits"
            ].argmax(dim=1)

            confusion[
                int(labels.item()),
                int(predictions.item()),
            ] += 1

    print(confusion)

    print()
    print("=" * 70)
    print("BALANCED V1 OVERFIT TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()