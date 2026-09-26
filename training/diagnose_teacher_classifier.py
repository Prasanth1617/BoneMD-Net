import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Device:", DEVICE)

    checkpoint = torch.load(
        "checkpoints/teacher_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherV1().to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("\n")
    print("=" * 70)
    print("TEACHER CLASSIFIER WEIGHT DIAGNOSTIC")
    print("=" * 70)

    classifier = model.classifier

    # Final Linear layer: 256 -> 3
    final_layer = classifier[-1]

    print("\nFinal classifier layer:")
    print("Weight shape:", tuple(final_layer.weight.shape))
    print("Bias shape:", tuple(final_layer.bias.shape))

    print("\nFinal classifier weights:")
    print(final_layer.weight.detach().cpu())

    print("\nFinal classifier bias:")
    print(final_layer.bias.detach().cpu())

    print("\nWeight statistics by output class:")

    for class_idx in range(3):
        weights = final_layer.weight[class_idx].detach().cpu()
        bias = final_layer.bias[class_idx].detach().cpu()

        print(
            f"Class {class_idx}: "
            f"mean={weights.mean():.8f} | "
            f"std={weights.std():.8f} | "
            f"min={weights.min():.8f} | "
            f"max={weights.max():.8f} | "
            f"norm={torch.norm(weights):.8f} | "
            f"bias={bias.item():.8f}"
        )

    # ---------------------------------------------------------
    # Check earlier classifier layer
    # ---------------------------------------------------------

    first_layer = classifier[0]

    print("\n")
    print("=" * 70)
    print("FIRST CLASSIFIER LAYER")
    print("=" * 70)

    print("Weight shape:", tuple(first_layer.weight.shape))
    print("Bias shape:", tuple(first_layer.bias.shape))

    print("\nWeight mean:", float(first_layer.weight.mean()))
    print("Weight std:", float(first_layer.weight.std()))
    print("Weight norm:", float(torch.norm(first_layer.weight)))

    print("\nBias mean:", float(first_layer.bias.mean()))
    print("Bias std:", float(first_layer.bias.std()))
    print("Bias norm:", float(torch.norm(first_layer.bias)))

    # ---------------------------------------------------------
    # Checkpoint metadata
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CHECKPOINT INFORMATION")
    print("=" * 70)

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))

        for key in checkpoint:
            if key != "model_state_dict":
                value = checkpoint[key]

                if isinstance(value, (int, float, str)):
                    print(f"{key}: {value}")

    print("\n")
    print("=" * 70)
    print("CLASSIFIER DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()