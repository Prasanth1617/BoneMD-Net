import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_norm import BoneMDTeacherNorm


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "FINAL_BEST_TEACHER_75_61.pth"
)

TEST_CACHE = (
    PROJECT_ROOT
    / "cache"
    / "test"
)

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


class BoneMDInferenceEngine:

    def __init__(self):

        self.device = DEVICE

        self.model = BoneMDTeacherNorm(
            feature_dim=512,
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

        self.best_validation_accuracy = checkpoint.get(
            "val_accuracy",
            "unknown",
        )

        self.dataset = CachedMultimodalDataset(
            TEST_CACHE
        )

    def find_patient(self, patient_id):

        patient_id = int(patient_id)

        for index in range(len(self.dataset)):

            sample = self.dataset[index]

            if int(sample["patient_id"]) == patient_id:
                return sample

        raise ValueError(
            f"Patient {patient_id} was not found "
            f"in the test cache."
        )

    def predict_patient(self, patient_id):

        sample = self.find_patient(
            patient_id
        )

        ap = (
            sample["ap"]
            .unsqueeze(0)
            .to(self.device)
        )

        lateral = (
            sample["lateral"]
            .unsqueeze(0)
            .to(self.device)
        )

        ct = (
            sample["ct"]
            .unsqueeze(0)
            .to(self.device)
        )

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

        return {
            "patient_id": int(
                sample["patient_id"]
            ),

            "true_label": int(
                sample["label"]
            ),

            "true_name": CLASS_NAMES[
                int(sample["label"])
            ],

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

            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),

            "best_validation_accuracy": (
                self.best_validation_accuracy
            ),

            "ap_shape": tuple(
                sample["ap"].shape
            ),

            "lateral_shape": tuple(
                sample["lateral"].shape
            ),

            "ct_shape": tuple(
                sample["ct"].shape
            ),

            "ap_features": tuple(
                outputs["ap_features"].shape
            ),

            "lateral_features": tuple(
                outputs["lateral_features"].shape
            ),

            "ct_features": tuple(
                outputs["ct_features"].shape
            ),

            "fused_features": tuple(
                outputs["fused_features"].shape
            ),

            "ap_tensor": sample["ap"],

            "lateral_tensor": sample["lateral"],

            "ct_tensor": sample["ct"],
        }