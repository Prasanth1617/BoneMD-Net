import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stats(name, tensor):
    tensor = tensor.detach()

    print(f"\n{name}")
    print("-" * 60)
    print("Shape:", tuple(tensor.shape))
    print("Mean:", float(tensor.mean()))
    print("Std:", float(tensor.std()))
    print("Min:", float(tensor.min()))
    print("Max:", float(tensor.max()))

    norms = torch.norm(tensor, dim=1)

    print("Mean norm:", float(norms.mean()))
    print("Std norm:", float(norms.std()))


def main():
    print("Device:", DEVICE)

    dataset = CachedMultimodalDataset("cache/train")

    checkpoint = torch.load(
        PROJECT_ROOT
        / "checkpoints"
        / "teacher_contrastive_queue_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model = BoneMDTeacherContrastive(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    ap_features_all = []
    lateral_features_all = []
    ct_features_all = []
    fused_features_all = []

    print("Samples:", len(dataset))

    with torch.no_grad():

        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            output = model(ap, lateral, ct)

            ap_features_all.append(
                output["ap_features"].cpu()
            )

            lateral_features_all.append(
                output["lateral_features"].cpu()
            )

            ct_features_all.append(
                output["ct_features"].cpu()
            )

            fused_features_all.append(
                output["fused_features"].cpu()
            )

    ap_features = torch.cat(ap_features_all, dim=0)
    lateral_features = torch.cat(lateral_features_all, dim=0)
    ct_features = torch.cat(ct_features_all, dim=0)
    fused_features = torch.cat(fused_features_all, dim=0)

    stats("AP FEATURES", ap_features)
    stats("LATERAL FEATURES", lateral_features)
    stats("CT FEATURES", ct_features)
    stats("FUSED FEATURES", fused_features)

    # ---------------------------------------------------------
    # Fusion layer weight analysis
    # ---------------------------------------------------------

    fusion_linear1 = model.fusion[0]

    weight = fusion_linear1.weight.detach().cpu()

    ap_weight = weight[:, 0:512]
    lateral_weight = weight[:, 512:1024]
    ct_weight = weight[:, 1024:1536]

    print("\n")
    print("=" * 60)
    print("FUSION FIRST-LAYER WEIGHT ANALYSIS")
    print("=" * 60)

    print("\nWeight absolute means:")
    print("AP:      ", float(ap_weight.abs().mean()))
    print("Lateral: ", float(lateral_weight.abs().mean()))
    print("CT:      ", float(ct_weight.abs().mean()))

    print("\nWeight Frobenius norms:")
    print("AP:      ", float(torch.norm(ap_weight)))
    print("Lateral: ", float(torch.norm(lateral_weight)))
    print("CT:      ", float(torch.norm(ct_weight)))

    # ---------------------------------------------------------
    # Estimate contribution magnitude
    # ---------------------------------------------------------

    ap_contribution = ap_features @ ap_weight.T
    lateral_contribution = lateral_features @ lateral_weight.T
    ct_contribution = ct_features @ ct_weight.T

    stats("AP -> FUSION CONTRIBUTION", ap_contribution)
    stats("LATERAL -> FUSION CONTRIBUTION", lateral_contribution)
    stats("CT -> FUSION CONTRIBUTION", ct_contribution)

    print("\n")
    print("=" * 60)
    print("FUSION DIAGNOSTIC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()