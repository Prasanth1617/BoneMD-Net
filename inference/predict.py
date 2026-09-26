import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_student import BoneMDStudent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_kd_v2_best.pth"
TEST_CACHE = PROJECT_ROOT / "cache" / "test"

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


def load_model():
    model = BoneMDStudent(
        feature_dim=256,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
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
        outputs = model(ap, lateral, ct)

        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

    return predicted_class, probabilities[0].cpu()


def main():
    parser = argparse.ArgumentParser(
        description="BoneMD-Net single-patient inference"
    )

    parser.add_argument(
        "--patient",
        type=int,
        required=True,
        help="LUMOS patient ID from the test set",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BoneMD-Net SINGLE-PATIENT INFERENCE")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Patient requested: {args.patient}")
    print(f"Checkpoint: {CHECKPOINT}")

    dataset = CachedMultimodalDataset(TEST_CACHE)

    print(f"Test cache samples: {len(dataset)}")

    model, checkpoint = load_model()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    sample = find_patient(
        dataset,
        args.patient,
    )

    patient_id = int(sample["patient_id"])
    true_label = int(sample["label"])

    predicted_class, probabilities = predict_patient(
        model,
        sample,
    )

    print()
    print(f"Patient ID: {patient_id}")
    print(f"True label: {true_label} ({CLASS_NAMES[true_label]})")
    print(
        f"Predicted: {predicted_class} "
        f"({CLASS_NAMES[predicted_class]})"
    )

    print()
    print("Class probabilities:")

    for class_id in range(3):
        probability = probabilities[class_id].item() * 100

        print(
            f"  {CLASS_NAMES[class_id]:15s}: "
            f"{probability:6.2f}%"
        )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
