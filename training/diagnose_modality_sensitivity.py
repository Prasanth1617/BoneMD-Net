import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_logits(name, logits):
    probs = torch.softmax(logits, dim=1)

    print(f"\n{name}")
    print("-" * 60)
    print("Logits:", [round(float(x), 6) for x in logits[0]])
    print("Probs: ", [round(float(x), 6) for x in probs[0]])


def main():
    print("Device:", DEVICE)

    dataset = CachedMultimodalDataset("cache/train")

    # Use one patient from the cached training set.
    sample = dataset[0]

    ap = sample["ap"].unsqueeze(0).to(DEVICE)
    lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
    ct = sample["ct"].unsqueeze(0).to(DEVICE)

    print("Patient ID:", sample["patient_id"])
    print("True label:", int(sample["label"]))

    checkpoint_path = "checkpoints/teacher_best.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherV1().to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():

        # ---------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------
        baseline = model(ap, lateral, ct)
        baseline_logits = baseline["logits"]

        print_logits("BASELINE: AP + LATERAL + CT", baseline_logits)

        # ---------------------------------------------------------
        # Change AP only
        # ---------------------------------------------------------
        if len(dataset) > 1:
            sample2 = dataset[1]
            ap2 = sample2["ap"].unsqueeze(0).to(DEVICE)

            ap_changed = model(ap2, lateral, ct)
            ap_changed_logits = ap_changed["logits"]

            print_logits("AP CHANGED ONLY", ap_changed_logits)

            ap_difference = torch.abs(
                ap_changed_logits - baseline_logits
            )

            print(
                "Absolute logit change:",
                [round(float(x), 6) for x in ap_difference[0]],
            )
            print(
                "Mean absolute change:",
                round(float(ap_difference.mean()), 6),
            )

        # ---------------------------------------------------------
        # Change lateral only
        # ---------------------------------------------------------
        if len(dataset) > 2:
            sample3 = dataset[2]
            lateral3 = sample3["lateral"].unsqueeze(0).to(DEVICE)

            lateral_changed = model(ap, lateral3, ct)
            lateral_changed_logits = lateral_changed["logits"]

            print_logits(
                "LATERAL CHANGED ONLY",
                lateral_changed_logits,
            )

            lateral_difference = torch.abs(
                lateral_changed_logits - baseline_logits
            )

            print(
                "Absolute logit change:",
                [round(float(x), 6) for x in lateral_difference[0]],
            )
            print(
                "Mean absolute change:",
                round(float(lateral_difference.mean()), 6),
            )

        # ---------------------------------------------------------
        # Change CT only
        # ---------------------------------------------------------
        if len(dataset) > 3:
            sample4 = dataset[3]
            ct4 = sample4["ct"].unsqueeze(0).to(DEVICE)

            ct_changed = model(ap, lateral, ct4)
            ct_changed_logits = ct_changed["logits"]

            print_logits("CT CHANGED ONLY", ct_changed_logits)

            ct_difference = torch.abs(
                ct_changed_logits - baseline_logits
            )

            print(
                "Absolute logit change:",
                [round(float(x), 6) for x in ct_difference[0]],
            )
            print(
                "Mean absolute change:",
                round(float(ct_difference.mean()), 6),
            )

    print("\n")
    print("=" * 60)
    print("MODALITY SENSITIVITY TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()