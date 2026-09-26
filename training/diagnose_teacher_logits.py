import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/teacher_best_v2.pth"

dataset = CachedMultimodalDataset("cache/train")

model = BoneMDTeacher(
    feature_dim=512,
    num_classes=3
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Device:", DEVICE)
print("Samples:", len(dataset))
print()

# Examine first 15 training patients
with torch.no_grad():
    for i in range(min(15, len(dataset))):

        sample = dataset[i]

        ap = sample["ap"].unsqueeze(0).to(DEVICE)
        lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
        ct = sample["ct"].unsqueeze(0).to(DEVICE)

        label = sample["label"].item()
        patient_id = sample["patient_id"]

        output = model(ap, lateral, ct)

        logits = output["logits"]
        probabilities = F.softmax(logits, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()

        print(
            f"Patient {patient_id:3d} | "
            f"Actual={label} | "
            f"Pred={prediction} | "
            f"Logits={logits.squeeze(0).cpu().numpy()} | "
            f"Prob={probabilities.squeeze(0).cpu().numpy()}"
        )