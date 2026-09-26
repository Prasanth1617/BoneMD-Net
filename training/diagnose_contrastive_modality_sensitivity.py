import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CACHE_DIR = PROJECT_ROOT / "cache" / "train"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)

CLASS_NAMES = [
    "Normal",
    "Osteopenia",
    "Osteoporosis",
]


def evaluate(model, loader, mode):
    correct = 0
    total = 0

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            if mode == "full":
                output = model(ap, lateral, ct)

            elif mode == "ap_only":
                output = model(
                    ap,
                    torch.zeros_like(lateral),
                    torch.zeros_like(ct),
                )

            elif mode == "lateral_only":
                output = model(
                    torch.zeros_like(ap),
                    lateral,
                    torch.zeros_like(ct),
                )

            elif mode == "ct_only":
                output = model(
                    torch.zeros_like(ap),
                    torch.zeros_like(lateral),
                    ct,
                )

            else:
                raise ValueError(mode)

            predicted = output["logits"].argmax(dim=1)

            correct += (
                predicted == batch["label"].to(DEVICE)
            ).sum().item()

            total += len(predicted)

            predictions.extend(
                predicted.cpu().numpy().tolist()
            )

            labels.extend(
                batch["label"].numpy().tolist()
            )

    accuracy = correct / total

    return accuracy, np.asarray(labels), np.asarray(predictions)


def main():

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    dataset = CachedMultimodalDataset(CACHE_DIR)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Training samples:", len(dataset))

    model = BoneMDTeacherContrastive(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        checkpoint.get("val_accuracy"),
    )

    print()
    print("=" * 70)
    print("CONTRASTIVE TEACHER MODALITY SENSITIVITY")
    print("=" * 70)

    results = {}

    for mode in [
        "full",
        "ap_only",
        "lateral_only",
        "ct_only",
    ]:

        accuracy, labels, predictions = evaluate(
            model,
            loader,
            mode,
        )

        results[mode] = accuracy

        print()
        print(
            f"{mode:15s} accuracy = "
            f"{accuracy:.4f}"
        )

        print("Predicted distribution:")

        for class_id, class_name in enumerate(CLASS_NAMES):

            count = int(
                np.sum(predictions == class_id)
            )

            print(
                f"  {class_id} ({class_name}): "
                f"{count}"
            )

    print()
    print("=" * 70)
    print("MODALITY ACCURACY COMPARISON")
    print("=" * 70)

    for mode, accuracy in results.items():

        print(
            f"{mode:15s}: "
            f"{accuracy:.4f}"
        )

    print()
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()