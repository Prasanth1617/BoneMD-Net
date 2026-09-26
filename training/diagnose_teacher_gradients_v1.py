import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "checkpoints/teacher_v1_baseline_best.pth"


def main():

    print("Device:", DEVICE)
    print()
    print("=" * 70)
    print("V1 BASELINE GRADIENT DIAGNOSTIC")
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherV1(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.train()

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint["epoch"])

    dataset = CachedMultimodalDataset("cache/train")

    print("Training samples:", len(dataset))

    criterion = nn.CrossEntropyLoss()

    print()
    print("=" * 70)
    print("PER-SAMPLE GRADIENT ANALYSIS")
    print("=" * 70)

    samples_per_class = {
        0: None,
        1: None,
        2: None,
    }

    for index in range(len(dataset)):

        sample = dataset[index]
        label = int(sample["label"])

        if samples_per_class[label] is None:
            samples_per_class[label] = index

        if all(
            value is not None
            for value in samples_per_class.values()
        ):
            break

    for class_id in range(3):

        index = samples_per_class[class_id]
        sample = dataset[index]

        ap = sample["ap"].unsqueeze(0).to(DEVICE)
        lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
        ct = sample["ct"].unsqueeze(0).to(DEVICE)

        label = torch.tensor(
            [class_id],
            dtype=torch.long,
            device=DEVICE,
        )

        model.zero_grad(set_to_none=True)

        output = model(
            ap,
            lateral,
            ct,
        )

        logits = output["logits"]

        loss = criterion(
            logits,
            label,
        )

        loss.backward()

        print()
        print(f"TRUE CLASS {class_id}")
        print("-" * 70)

        print("Patient ID:", sample["patient_id"])

        print(
            "Logits:",
            [
                round(float(x), 6)
                for x in logits[0].detach().cpu()
            ],
        )

        probabilities = torch.softmax(
            logits[0].detach(),
            dim=0,
        )

        print(
            "Probabilities:",
            [
                round(float(x), 6)
                for x in probabilities.cpu()
            ],
        )

        print(
            "Loss:",
            round(float(loss.detach().item()), 6),
        )

        final_weight_grad = (
            model.classifier[-1].weight.grad
        )

        final_bias_grad = (
            model.classifier[-1].bias.grad
        )

        print()
        print("Final classifier weight gradient:")
        print(
            "  Shape:",
            tuple(final_weight_grad.shape),
        )
        print(
            "  Mean abs:",
            float(
                final_weight_grad.abs().mean().item()
            ),
        )
        print(
            "  Norm:",
            float(
                final_weight_grad.norm().item()
            ),
        )

        print("Final classifier bias gradient:")
        print(
            [
                round(float(x), 6)
                for x in final_bias_grad.detach().cpu()
            ]
        )

        fusion_weight_grad = (
            model.fusion[0].weight.grad
        )

        print()
        print("Fusion first-layer gradient:")
        print(
            "  Mean abs:",
            float(
                fusion_weight_grad.abs().mean().item()
            ),
        )
        print(
            "  Norm:",
            float(
                fusion_weight_grad.norm().item()
            ),
        )

        ap_grad = model.ap_encoder.projection.weight.grad
        lateral_grad = model.lateral_encoder.projection.weight.grad
        ct_grad = model.ct_encoder.projection.weight.grad

        print()
        print("Encoder projection gradient norms:")
        print(
            "  AP:      ",
            float(ap_grad.norm().item()),
        )
        print(
            "  Lateral: ",
            float(lateral_grad.norm().item()),
        )
        print(
            "  CT:      ",
            float(ct_grad.norm().item()),
        )

    print()
    print("=" * 70)
    print("V1 BASELINE GRADIENT DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()