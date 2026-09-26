import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "teacher_contrastive_queue_best.pth"
TRAIN_CACHE = PROJECT_ROOT / "cache" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(name, tensor):
    x = tensor.detach().float()

    # Flatten every sample except batch dimension.
    x = x.reshape(x.shape[0], -1)

    sample_std = x.std(dim=0)

    print()
    print(f"{name}")
    print("-" * 70)
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Mean absolute value: {x.abs().mean().item():.8f}")
    print(f"Mean feature std:    {sample_std.mean().item():.8f}")
    print(f"Median feature std:  {sample_std.median().item():.8f}")
    print(f"Max feature std:     {sample_std.max().item():.8f}")
    print(f"Sample norm mean:     {x.norm(dim=1).mean().item():.8f}")
    print(f"Sample norm std:      {x.norm(dim=1).std().item():.8f}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Cache: {TRAIN_CACHE}")

    dataset = CachedMultimodalDataset(str(TRAIN_CACHE))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherContrastive(
        num_classes=3,
        feature_dim=512,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")

    # We inspect the AP encoder only.
    encoder = model.ap_encoder

    # Locate the feature-producing stages.
    captured = {}

    def hook(name):
        def _hook(module, inputs, output):
            captured[name] = output.detach().cpu()

        return _hook

    # Register hooks on every major module in the encoder.
    for idx, module in enumerate(encoder.features):
        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.GroupNorm,
                nn.BatchNorm2d,
                nn.ReLU,
                nn.AdaptiveAvgPool2d,
            ),
        ):
            module.register_forward_hook(
                hook(f"features[{idx}] {module.__class__.__name__}")
            )

    all_inputs = []

    # Use the first 190 training samples.
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)

            # Forward through the encoder.
            _ = encoder(ap)

            # Store the captured outputs for this patient.
            if i == 0:
                for name, value in captured.items():
                    all_inputs.append(
                        {
                            "name": name,
                            "values": [value.squeeze(0)],
                        }
                    )
            else:
                for item in all_inputs:
                    item["values"].append(
                        captured[item["name"]].squeeze(0)
                    )

    print()
    print("=" * 70)
    print("AP ENCODER LAYER-BY-LAYER VARIABILITY")
    print("=" * 70)

    for item in all_inputs:
        tensor = torch.stack(item["values"], dim=0)
        summarize(item["name"], tensor)

    # Also inspect final projection output before normalization.
    raw_projection = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)

            x = model.ap_encoder.features(ap)
            x = torch.flatten(x, 1)
            x = model.ap_encoder.projection(x)

            raw_projection.append(
                x.squeeze(0).cpu()
            )

    raw_projection = torch.stack(raw_projection)

    summarize(
        "AP projection output (before normalization)",
        raw_projection,
    )


if __name__ == "__main__":
    main()