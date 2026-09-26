import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import glob


for split in ["train", "val"]:

    print()
    print("=" * 60)
    print(f"{split.upper()} CLASS STATISTICS")
    print("=" * 60)

    stats = {
        0: {"ap": [], "lateral": [], "ct": []},
        1: {"ap": [], "lateral": [], "ct": []},
        2: {"ap": [], "lateral": [], "ct": []},
    }

    files = sorted(
        glob.glob(f"cache/{split}/patient_*.pt")
    )

    for path in files:

        sample = torch.load(
            path,
            map_location="cpu"
        )

        label = int(sample["label"].item())

        for modality in ["ap", "lateral", "ct"]:

            tensor = sample[modality].float()

            stats[label][modality].append({
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
            })

    for cls in range(3):

        print()
        print(f"CLASS {cls}")

        for modality in ["ap", "lateral", "ct"]:

            values = stats[cls][modality]

            means = [x["mean"] for x in values]
            stds = [x["std"] for x in values]

            print(
                f"{modality.upper():8s} "
                f"samples={len(values):3d} | "
                f"mean={sum(means)/len(means):.4f} | "
                f"std={sum(stds)/len(stds):.4f}"
            )