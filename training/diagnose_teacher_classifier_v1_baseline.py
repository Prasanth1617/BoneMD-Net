import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    "checkpoints/teacher_v1_baseline_best.pth"
)


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("V1 BASELINE CLASSIFIER DIAGNOSTIC")
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherV1(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint["epoch"])

    if "metrics" in checkpoint:
        print()
        print("Checkpoint metrics:")

        for key, value in checkpoint["metrics"].items():
            print(f"{key}: {value}")

    dataset = CachedMultimodalDataset(
        "cache/val"
    )

    all_labels = []
    all_predictions = []
    all_logits = []
    all_probabilities = []

    with torch.no_grad():

        for index in range(len(dataset)):

            sample = dataset[index]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            label = int(sample["label"])

            output = model(
                ap,
                lateral,
                ct,
            )

            logits = output["logits"][0]

            probabilities = torch.softmax(
                logits,
                dim=0,
            )

            prediction = int(
                logits.argmax().item()
            )

            all_labels.append(label)
            all_predictions.append(prediction)
            all_logits.append(
                logits.cpu()
            )
            all_probabilities.append(
                probabilities.cpu()
            )

    labels = torch.tensor(all_labels)
    predictions = torch.tensor(all_predictions)
    logits = torch.stack(all_logits)
    probabilities = torch.stack(
        all_probabilities
    )

    print()
    print("=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    for class_id in range(3):

        true_count = int(
            (labels == class_id).sum()
        )

        predicted_count = int(
            (predictions == class_id).sum()
        )

        print(
            f"Class {class_id}: "
            f"true={true_count} | "
            f"predicted={predicted_count}"
        )

    accuracy = (
        predictions == labels
    ).float().mean()

    print()
    print("Validation accuracy:", f"{float(accuracy) * 100:.2f}%")

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    confusion = torch.zeros(
        3,
        3,
        dtype=torch.long,
    )

    for true, pred in zip(
        labels,
        predictions,
    ):
        confusion[
            int(true),
            int(pred),
        ] += 1

    print(confusion)

    print()
    print("=" * 70)
    print("MEAN LOGITS BY TRUE CLASS")
    print("=" * 70)

    for class_id in range(3):

        mask = labels == class_id

        class_logits = logits[mask]

        mean_logits = class_logits.mean(
            dim=0
        )

        std_logits = class_logits.std(
            dim=0
        )

        print(
            f"Class {class_id}:"
        )

        print(
            "  Mean:",
            [
                round(float(x), 6)
                for x in mean_logits
            ],
        )

        print(
            "  Std: ",
            [
                round(float(x), 6)
                for x in std_logits
            ],
        )

    print()
    print("=" * 70)
    print("MEAN PROBABILITIES BY TRUE CLASS")
    print("=" * 70)

    for class_id in range(3):

        mask = labels == class_id

        class_probabilities = (
            probabilities[mask]
        )

        mean_probabilities = (
            class_probabilities.mean(dim=0)
        )

        print(
            f"Class {class_id}:"
        )

        print(
            "  Mean:",
            [
                round(float(x), 6)
                for x in mean_probabilities
            ],
        )

    print()
    print("=" * 70)
    print("V1 BASELINE CLASSIFIER DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()