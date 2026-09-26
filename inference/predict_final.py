import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_norm import BoneMDTeacherNorm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "FINAL_BEST_TEACHER_75_61.pth"
)

TEST_CACHE = PROJECT_ROOT / "cache" / "test"

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


def load_model():
    model = BoneMDTeacherNorm(
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

    return model, checkpoint


def find_patient(dataset, patient_id):
    for index in range(len(dataset)):
        sample = dataset[index]

        if int(sample["patient_id"]) == patient_id:
            return sample

    raise ValueError(
        f"Patient {patient_id} was not found in the test cache."
    )


def predict_patient(model, sample):
    ap = sample["ap"].unsqueeze(0).to(DEVICE)
    lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
    ct = sample["ct"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            ap,
            lateral,
            ct,
        )

        logits = outputs["logits"]

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

    return (
        predicted_class,
        probabilities[0].cpu(),
        outputs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="BoneMD-Net final teacher single-patient inference"
    )

    parser.add_argument(
        "--patient",
        type=int,
        required=True,
        help="LUMOS patient ID from the test cache",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BoneMD-Net FINAL MODEL INFERENCE")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Patient requested: {args.patient}")
    print(f"Checkpoint: {CHECKPOINT}")

    dataset = CachedMultimodalDataset(
        TEST_CACHE
    )

    print(f"Test cache samples: {len(dataset)}")

    model, checkpoint = load_model()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint.get('val_accuracy', 'unknown'):.4f}"
    )

    sample = find_patient(
        dataset,
        args.patient,
    )

    patient_id = int(sample["patient_id"])
    true_label = int(sample["label"])

    print()
    print("Input tensors")
    print("-" * 70)

    print(
        f"AP X-ray:      {tuple(sample['ap'].shape)}"
    )

    print(
        f"Lateral X-ray: {tuple(sample['lateral'].shape)}"
    )

    print(
        f"CT volume:     {tuple(sample['ct'].shape)}"
    )

    predicted_class, probabilities, outputs = (
        predict_patient(
            model,
            sample,
        )
    )

    print()
    print("Prediction")
    print("-" * 70)

    print(
        f"Patient ID: {patient_id}"
    )

    print(
        f"True label: {true_label} "
        f"({CLASS_NAMES[true_label]})"
    )

    print(
        f"Predicted:  {predicted_class} "
        f"({CLASS_NAMES[predicted_class]})"
    )

    print()
    print("Class probabilities")
    print("-" * 70)

    for class_id in range(3):
        probability = (
            probabilities[class_id].item()
            * 100
        )

        print(
            f"  {CLASS_NAMES[class_id]:15s}: "
            f"{probability:6.2f}%"
        )

    print()
    print("Feature dimensions")
    print("-" * 70)

    print(
        f"AP features:       "
        f"{tuple(outputs['ap_features'].shape)}"
    )

    print(
        f"Lateral features:  "
        f"{tuple(outputs['lateral_features'].shape)}"
    )

    print(
        f"CT features:       "
        f"{tuple(outputs['ct_features'].shape)}"
    )

    print(
        f"Fused features:    "
        f"{tuple(outputs['fused_features'].shape)}"
    )

    print()
    print("=" * 70)
    print("FINAL MODEL INFERENCE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
