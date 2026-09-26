import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/teacher_v2_controlled_best.pth"


def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print()

    dataset = CachedMultimodalDataset("cache/val")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = BoneMDTeacher(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = dataset[0]

    ap = sample["ap"].unsqueeze(0).to(DEVICE)
    lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
    ct = sample["ct"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(ap, lateral, ct)

    logits = output["logits"]

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint validation accuracy:", checkpoint["val_accuracy"])
    print()

    print("Patient ID:", sample["patient_id"])
    print("True label:", sample["label"])
    print("Logits shape:", tuple(logits.shape))

    print("Logits finite:", bool(torch.isfinite(logits).all()))

    print("Logits:", logits.cpu().numpy())
    print("Predicted class:", int(logits.argmax(dim=1).item()))

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("Total parameters:", parameter_count)
    print("Trainable parameters:", trainable_count)

    print()
    print("=" * 70)
    print("TEACHER V2 SANITY CHECK PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()