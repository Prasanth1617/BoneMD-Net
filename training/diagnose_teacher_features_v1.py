import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_v1 import BoneMDTeacherV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(name, features, labels):
    print(f"\n{name}")
    print("-" * 70)

    for label in [0, 1, 2]:
        mask = labels == label
        x = features[mask]

        norms = torch.norm(x, dim=1)

        print(
            f"Class {label}: "
            f"N={x.shape[0]} | "
            f"mean={x.mean():.6f} | "
            f"std={x.std():.6f} | "
            f"mean_norm={norms.mean():.6f} | "
            f"norm_std={norms.std():.6f}"
        )


def main():
    print("Device:", DEVICE)

    dataset = CachedMultimodalDataset("cache/train")

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

    ap_all = []
    lateral_all = []
    ct_all = []
    fused_all = []
    labels_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            output = model(ap, lateral, ct)

            ap_all.append(output["ap_features"].cpu())
            lateral_all.append(output["lateral_features"].cpu())
            ct_all.append(output["ct_features"].cpu())
            fused_all.append(output["fused_features"].cpu())
            labels_all.append(sample["label"].reshape(1))

    ap_features = torch.cat(ap_all, dim=0)
    lateral_features = torch.cat(lateral_all, dim=0)
    ct_features = torch.cat(ct_all, dim=0)
    fused_features = torch.cat(fused_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    print("\nSamples:", len(labels))
    print("Class counts:", {
        int(c): int((labels == c).sum())
        for c in [0, 1, 2]
    })

    summarize("AP FEATURES BY CLASS", ap_features, labels)
    summarize("LATERAL FEATURES BY CLASS", lateral_features, labels)
    summarize("CT FEATURES BY CLASS", ct_features, labels)
    summarize("FUSED FEATURES BY CLASS", fused_features, labels)

    # ---------------------------------------------------------
    # Logit statistics by class
    # ---------------------------------------------------------

    logits_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            ap = sample["ap"].unsqueeze(0).to(DEVICE)
            lateral = sample["lateral"].unsqueeze(0).to(DEVICE)
            ct = sample["ct"].unsqueeze(0).to(DEVICE)

            output = model(ap, lateral, ct)
            logits_all.append(output["logits"].cpu())

    logits = torch.cat(logits_all, dim=0)

    print("\n")
    print("=" * 70)
    print("LOGITS BY TRUE CLASS")
    print("=" * 70)

    for label in [0, 1, 2]:
        mask = labels == label
        class_logits = logits[mask]

        print(f"\nTrue class {label}")
        print("Mean logits:",
              [round(float(x), 6) for x in class_logits.mean(dim=0)])
        print("Std logits: ",
              [round(float(x), 6) for x in class_logits.std(dim=0)])

    print("\n")
    print("=" * 70)
    print("V1 TEACHER FEATURE DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()