import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


def collect_modality_outputs(model, loader):

    model.eval()

    records = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            # Full multimodal model
            full_output = model(
                ap,
                lateral,
                ct,
            )

            # Individual modality contribution at fusion input.
            ap_features = full_output["ap_features"]
            lateral_features = full_output["lateral_features"]
            ct_features = full_output["ct_features"]

            # Construct fusion inputs.
            full_fusion_input = torch.cat(
                [
                    ap_features,
                    lateral_features,
                    ct_features,
                ],
                dim=1,
            )

            ap_only_input = torch.cat(
                [
                    ap_features,
                    torch.zeros_like(lateral_features),
                    torch.zeros_like(ct_features),
                ],
                dim=1,
            )

            lateral_only_input = torch.cat(
                [
                    torch.zeros_like(ap_features),
                    lateral_features,
                    torch.zeros_like(ct_features),
                ],
                dim=1,
            )

            ct_only_input = torch.cat(
                [
                    torch.zeros_like(ap_features),
                    torch.zeros_like(lateral_features),
                    ct_features,
                ],
                dim=1,
            )

            def get_hidden(fusion_input):

                fused = model.fusion(
                    fusion_input
                )

                hidden = model.classifier[0](
                    fused
                )

                hidden = F.relu(
                    hidden
                )

                return hidden

            full_hidden = get_hidden(
                full_fusion_input
            )

            ap_hidden = get_hidden(
                ap_only_input
            )

            lateral_hidden = get_hidden(
                lateral_only_input
            )

            ct_hidden = get_hidden(
                ct_only_input
            )

            records.append(
                {
                    "label": batch["label"].cpu(),
                    "full": full_hidden.cpu(),
                    "ap": ap_hidden.cpu(),
                    "lateral": lateral_hidden.cpu(),
                    "ct": ct_hidden.cpu(),
                }
            )

    labels = torch.cat(
        [r["label"] for r in records]
    )

    full = torch.cat(
        [r["full"] for r in records]
    )

    ap = torch.cat(
        [r["ap"] for r in records]
    )

    lateral = torch.cat(
        [r["lateral"] for r in records]
    )

    ct = torch.cat(
        [r["ct"] for r in records]
    )

    return (
        labels,
        full,
        ap,
        lateral,
        ct,
    )


def main():

    print("Device:", DEVICE)

    model = BoneMDTeacherContrastive(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"],
    )

    print()

    dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "train"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    (
        labels,
        full,
        ap,
        lateral,
        ct,
    ) = collect_modality_outputs(
        model,
        loader,
    )

    print(
        "Samples:",
        len(labels),
    )

    print(
        "Hidden dimension:",
        full.shape[1],
    )

    print()

    # ------------------------------------------------------------
    # Overall modality contribution
    # ------------------------------------------------------------

    print("=" * 75)
    print("OVERALL HIDDEN REPRESENTATION NORMS")
    print("=" * 75)

    for name, features in [
        ("Full", full),
        ("AP only", ap),
        ("Lateral only", lateral),
        ("CT only", ct),
    ]:

        norms = torch.linalg.vector_norm(
            features,
            dim=1,
        )

        print(
            f"{name:12s} "
            f"mean={norms.mean().item():.6f} "
            f"std={norms.std().item():.6f}"
        )

    print()

    # ------------------------------------------------------------
    # Class-specific modality contribution
    # ------------------------------------------------------------

    print("=" * 75)
    print("CLASS-SPECIFIC HIDDEN REPRESENTATION NORMS")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        mask = labels == class_id

        print()
        print(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]}"
        )

        for name, features in [
            ("Full", full),
            ("AP only", ap),
            ("Lateral only", lateral),
            ("CT only", ct),
        ]:

            class_features = features[
                mask
            ]

            norms = torch.linalg.vector_norm(
                class_features,
                dim=1,
            )

            print(
                f"  {name:10s} "
                f"mean={norms.mean().item():.6f} "
                f"std={norms.std().item():.6f}"
            )

    print()

    # ------------------------------------------------------------
    # Distance from full representation
    # ------------------------------------------------------------

    print("=" * 75)
    print("DISTANCE FROM FULL MULTIMODAL REPRESENTATION")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        mask = labels == class_id

        print()
        print(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]}"
        )

        full_class = full[mask]

        for name, features in [
            ("AP only", ap),
            ("Lateral only", lateral),
            ("CT only", ct),
        ]:

            modality_class = features[
                mask
            ]

            distances = torch.linalg.vector_norm(
                full_class - modality_class,
                dim=1,
            )

            print(
                f"  {name:10s} "
                f"mean distance="
                f"{distances.mean().item():.6f} "
                f"std="
                f"{distances.std().item():.6f}"
            )

    print()

    # ------------------------------------------------------------
    # Modality representation similarity
    # ------------------------------------------------------------

    print("=" * 75)
    print("MODALITY / FULL REPRESENTATION COSINE SIMILARITY")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        mask = labels == class_id

        full_class = F.normalize(
            full[mask],
            dim=1,
        )

        print()
        print(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]}"
        )

        for name, features in [
            ("AP only", ap),
            ("Lateral only", lateral),
            ("CT only", ct),
        ]:

            modality_class = F.normalize(
                features[mask],
                dim=1,
            )

            similarity = (
                full_class
                * modality_class
            ).sum(dim=1)

            print(
                f"  {name:10s} "
                f"mean cosine="
                f"{similarity.mean().item():.6f} "
                f"std="
                f"{similarity.std().item():.6f}"
            )

    print()

    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()