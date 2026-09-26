import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "teacher_contrastive_queue_best.pth"
TRAIN_CACHE = PROJECT_ROOT / "cache" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(name, tensor):
    tensor = tensor.detach().float().cpu()

    print()
    print(f"{name}")
    print("-" * 60)
    print(f"Shape:              {tuple(tensor.shape)}")
    print(f"Mean:               {tensor.mean().item():.8f}")
    print(f"Std:                {tensor.std().item():.8f}")
    print(f"Min:                {tensor.min().item():.8f}")
    print(f"Max:                {tensor.max().item():.8f}")

    if tensor.ndim >= 2:
        flat = tensor.flatten(1)

        sample_means = flat.mean(dim=1)
        sample_stds = flat.std(dim=1)

        print(f"Patient mean std:   {sample_means.std().item():.8f}")
        print(f"Patient norm mean:  {flat.norm(dim=1).mean().item():.8f}")
        print(f"Patient norm std:   {flat.norm(dim=1).std().item():.8f}")


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

    # ------------------------------------------------------------
    # Collect activations from every X-ray encoder stage.
    # ------------------------------------------------------------

    ap_stage_outputs = [[] for _ in range(5)]
    lateral_stage_outputs = [[] for _ in range(5)]

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)

            # Access the sequential feature blocks directly.
            ap_encoder = model.ap_encoder
            lateral_encoder = model.lateral_encoder

            x = ap

            for stage_idx, layer in enumerate(ap_encoder.features):
                x = layer(x)

                # Record after major convolution/activation stages.
                if stage_idx in [2, 5, 8, 11, 12]:
                    slot = [2, 5, 8, 11, 12].index(stage_idx)
                    ap_stage_outputs[slot].append(x.squeeze(0).cpu())

            x = lateral

            for stage_idx, layer in enumerate(lateral_encoder.features):
                x = layer(x)

                if stage_idx in [2, 5, 8, 11, 12]:
                    slot = [2, 5, 8, 11, 12].index(stage_idx)
                    lateral_stage_outputs[slot].append(x.squeeze(0).cpu())

    stage_names = [
        "After block 1",
        "After block 2",
        "After block 3",
        "After block 4",
        "After final pooling",
    ]

    print()
    print("=" * 70)
    print("AP ENCODER")
    print("=" * 70)

    for name, tensors in zip(stage_names, ap_stage_outputs):
        stacked = torch.stack(tensors)
        summarize(name, stacked)

    print()
    print("=" * 70)
    print("LATERAL ENCODER")
    print("=" * 70)

    for name, tensors in zip(stage_names, lateral_stage_outputs):
        stacked = torch.stack(tensors)
        summarize(name, stacked)


if __name__ == "__main__":
    main()