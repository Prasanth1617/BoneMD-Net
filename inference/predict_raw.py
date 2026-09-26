import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import torch

from utils.xray_loader import load_xray
from utils.xray_preprocess import preprocess_xray
from utils.ct_loader import load_ct_volume
from utils.ct_preprocess import preprocess_ct
from models.bone_md_student import BoneMDStudent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_kd_v2_best.pth"
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def prepare_xray(zip_path, dicom_path):
    image, ds = load_xray(
        zip_path,
        dicom_path,
    )

    image = preprocess_xray(image)

    tensor = torch.from_numpy(image).float()

    return tensor.unsqueeze(0), ds


def prepare_ct(zip_path, patient_folder):
    volume, metadata = load_ct_volume(
        zip_path,
        patient_folder,
    )

    volume = preprocess_ct(
        volume,
        metadata,
    )

    tensor = torch.from_numpy(volume).float()

    return tensor.unsqueeze(0), metadata


def predict(model, ap, lateral, ct):
    ap = ap.unsqueeze(0).to(DEVICE)
    lateral = lateral.unsqueeze(0).to(DEVICE)
    ct = ct.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            ap,
            lateral,
            ct,
        )

        probabilities = torch.softmax(
            outputs["logits"],
            dim=1,
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        fusion_weights = outputs[
            "fusion_weights"
        ][0].cpu()

    return (
        predicted_class,
        probabilities.cpu(),
        fusion_weights,
    )


def get_patient_record(patient_id):
    manifest = pd.read_csv(MANIFEST)

    matches = manifest[
        manifest["patient_id"] == patient_id
    ]

    if matches.empty:
        raise ValueError(
            f"Patient {patient_id} was not found "
            f"in the final multimodal manifest."
        )

    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest row for patient "
            f"{patient_id}, found {len(matches)}."
        )

    return matches.iloc[0]


def main():
    parser = argparse.ArgumentParser(
        description="BoneMD-Net raw DICOM inference"
    )

    parser.add_argument(
        "--patient",
        type=int,
        required=True,
        help="LUMOS patient ID",
    )

    args = parser.parse_args()

    patient_id = args.patient

    record = get_patient_record(patient_id)

    label = int(record["label"])

    ap_path = str(record["ap_dicom_file"])
    lateral_path = str(record["lateral_dicom_file"])
    ct_zip_name = str(record["ct_zip"])
    ct_folder = str(record["ct_patient_folder"])

    xray_zip = PROJECT_ROOT / "dataset" / "lumos_x_001_280_dcm.zip"
    ct_zip = PROJECT_ROOT / "dataset" / ct_zip_name

    print("=" * 70)
    print("BoneMD-Net RAW DICOM INFERENCE")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Patient: {patient_id}")
    print(f"Checkpoint: {CHECKPOINT}")

    print()
    print(
        f"True label: {label} "
        f"({CLASS_NAMES[label]})"
    )

    print()
    print("Loading AP X-ray...")

    ap, ap_ds = prepare_xray(
        xray_zip,
        ap_path,
    )

    print(
        f"AP tensor: {tuple(ap.shape)}"
    )

    print("Loading lateral X-ray...")

    lateral, lateral_ds = prepare_xray(
        xray_zip,
        lateral_path,
    )

    print(
        f"Lateral tensor: {tuple(lateral.shape)}"
    )

    print("Loading CT...")

    ct, metadata = prepare_ct(
        ct_zip,
        ct_folder,
    )

    print(
        f"CT tensor: {tuple(ct.shape)}"
    )

    model, checkpoint = load_model()

    print()
    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    predicted_class, probabilities, fusion_weights = predict(
        model,
        ap,
        lateral,
        ct,
    )

    print()
    print("Prediction")
    print("-" * 70)

    print(
        f"Predicted class: {predicted_class} "
        f"({CLASS_NAMES[predicted_class]})"
    )

    print()
    print("Class probabilities:")

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
    print("Fusion weights:")

    print(
        f"  AP:       {fusion_weights[0].item():.4f}"
    )

    print(
        f"  Lateral:  {fusion_weights[1].item():.4f}"
    )

    print(
        f"  CT:       {fusion_weights[2].item():.4f}"
    )

    print()
    print("=" * 70)
    print("RAW INFERENCE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
