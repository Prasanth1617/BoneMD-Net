import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)

class_names = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


def analyze_split(model, split):

    dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / split
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    records = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(ap, lateral, ct)

            logits = output["logits"]

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

            true_label = int(
                batch["label"][0]
            )

            predicted_label = int(
                prediction[0]
            )

            records.append(
                {
                    "patient_id": int(
                        batch["patient_id"][0]
                    ),
                    "true": true_label,
                    "pred": predicted_label,
                    "confidence": float(
                        confidence[0]
                    ),
                    "probabilities": (
                        probabilities[0]
                        .cpu()
                        .numpy()
                    ),
                }
            )

    return records


def print_split_analysis(records, split):

    print()
    print("=" * 75)
    print(f"{split.upper()} CLASSIFIER CONFIDENCE")
    print("=" * 75)

    correct = [
        r for r in records
        if r["true"] == r["pred"]
    ]

    incorrect = [
        r for r in records
        if r["true"] != r["pred"]
    ]

    accuracy = (
        len(correct) / len(records)
    )

    print("Samples:", len(records))
    print("Correct:", len(correct))
    print("Incorrect:", len(incorrect))
    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("OVERALL CONFIDENCE")
    print("-" * 75)

    all_conf = np.array(
        [r["confidence"] for r in records]
    )

    correct_conf = np.array(
        [r["confidence"] for r in correct]
    )

    incorrect_conf = np.array(
        [r["confidence"] for r in incorrect]
    )

    print(
        "Mean confidence:",
        all_conf.mean(),
    )

    if len(correct_conf):
        print(
            "Correct mean confidence:",
            correct_conf.mean(),
        )

    if len(incorrect_conf):
        print(
            "Incorrect mean confidence:",
            incorrect_conf.mean(),
        )

    print()
    print("PER-CLASS CONFIDENCE")
    print("-" * 75)

    for class_id in [0, 1, 2]:

        class_records = [
            r for r in records
            if r["true"] == class_id
        ]

        class_correct = [
            r for r in class_records
            if r["true"] == r["pred"]
        ]

        if not class_records:
            continue

        confidences = np.array(
            [
                r["confidence"]
                for r in class_records
            ]
        )

        print(
            f"{class_names[class_id]} "
            f"(n={len(class_records)}): "
            f"mean={confidences.mean():.4f}, "
            f"correct={len(class_correct)}/{len(class_records)}"
        )

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 75)

    for class_id in [0, 1, 2]:

        predicted_count = sum(
            r["pred"] == class_id
            for r in records
        )

        print(
            f"{class_names[class_id]}: "
            f"{predicted_count}"
        )

    print()
    print("INCORRECT PREDICTIONS")
    print("-" * 75)

    for r in incorrect:

        probs = r["probabilities"]

        print(
            f"Patient {r['patient_id']:03d}: "
            f"{class_names[r['true']]} -> "
            f"{class_names[r['pred']]} | "
            f"confidence={r['confidence']:.4f} | "
            f"P(N)={probs[0]:.4f} "
            f"P(Ope)={probs[1]:.4f} "
            f"P(Opo)={probs[2]:.4f}"
        )

    print()
    print("LOW-CONFIDENCE PREDICTIONS")
    print("-" * 75)

    sorted_records = sorted(
        records,
        key=lambda r: r["confidence"]
    )

    for r in sorted_records[:15]:

        status = (
            "CORRECT"
            if r["true"] == r["pred"]
            else "WRONG"
        )

        probs = r["probabilities"]

        print(
            f"Patient {r['patient_id']:03d}: "
            f"{class_names[r['true']]} -> "
            f"{class_names[r['pred']]} | "
            f"{status} | "
            f"confidence={r['confidence']:.4f} | "
            f"P(N)={probs[0]:.4f} "
            f"P(Ope)={probs[1]:.4f} "
            f"P(Opo)={probs[2]:.4f}"
        )


def main():

    print("Device:", DEVICE)

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

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint.get("val_accuracy"),
    )

    train_records = analyze_split(
        model,
        "train",
    )

    val_records = analyze_split(
        model,
        "val",
    )

    print_split_analysis(
        train_records,
        "train",
    )

    print_split_analysis(
        val_records,
        "val",
    )

    print()
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()