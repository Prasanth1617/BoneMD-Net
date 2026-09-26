import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.xray_loader import load_xray
from utils.xray_preprocess import preprocess_xray
from utils.ct_loader import load_ct_volume
from utils.ct_preprocess import preprocess_ct
from models.bone_md_student import BoneMDStudent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_kd_v2_best.pth"

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


class BoneMDInferenceEngine:

    def __init__(self):
        self.device = DEVICE

        self.model = BoneMDStudent(
            feature_dim=256,
            num_classes=3,
        ).to(self.device)

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.checkpoint_epoch = checkpoint.get(
            "epoch",
            "unknown",
        )

    def prepare_xray(self, zip_path, dicom_path):
        image, ds = load_xray(
            zip_path,
            dicom_path,
        )

        image = preprocess_xray(image)

        tensor = torch.from_numpy(
            image
        ).float()

        return tensor.unsqueeze(0), ds

    def prepare_ct(self, zip_path, patient_folder):
        volume, metadata = load_ct_volume(
            zip_path,
            patient_folder,
        )

        volume = preprocess_ct(
            volume,
            metadata,
        )

        tensor = torch.from_numpy(
            volume
        ).float()

        return tensor.unsqueeze(0), metadata

    def predict(
        self,
        ap_zip,
        ap_path,
        lateral_zip,
        lateral_path,
        ct_zip,
        ct_folder,
    ):
        ap, ap_ds = self.prepare_xray(
            ap_zip,
            ap_path,
        )

        lateral, lateral_ds = self.prepare_xray(
            lateral_zip,
            lateral_path,
        )

        ct, ct_metadata = self.prepare_ct(
            ct_zip,
            ct_folder,
        )

        ap = ap.unsqueeze(0).to(self.device)
        lateral = lateral.unsqueeze(0).to(self.device)
        ct = ct.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
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
            ][0]

        return {
            "predicted_class": predicted_class,
            "predicted_name": CLASS_NAMES[
                predicted_class
            ],
            "probabilities": {
                CLASS_NAMES[i]: float(
                    probabilities[i].item()
                )
                for i in range(3)
            },
            "fusion_weights": {
                "AP": float(
                    fusion_weights[0].item()
                ),
                "Lateral": float(
                    fusion_weights[1].item()
                ),
                "CT": float(
                    fusion_weights[2].item()
                ),
            },
            "checkpoint_epoch": self.checkpoint_epoch,
            "ap_shape": tuple(ap.shape),
            "lateral_shape": tuple(lateral.shape),
            "ct_shape": tuple(ct.shape),
        }
