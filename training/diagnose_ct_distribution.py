import glob
import torch


for split in ["train", "val"]:

    print()
    print("=" * 60)
    print(f"{split.upper()} CT DISTRIBUTION")
    print("=" * 60)

    stats = {
        0: [],
        1: [],
        2: [],
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

        ct = sample["ct"].float()

        total = ct.numel()

        # Percentage of voxels above air/background
        pct_above_minus900 = (
            (ct > -900).float().mean().item() * 100
        )

        pct_above_minus500 = (
            (ct > -500).float().mean().item() * 100
        )

        pct_above_0 = (
            (ct > 0).float().mean().item() * 100
        )

        stats[label].append(
            (
                pct_above_minus900,
                pct_above_minus500,
                pct_above_0,
            )
        )

    for cls in range(3):

        values = stats[cls]

        a = [x[0] for x in values]
        b = [x[1] for x in values]
        c = [x[2] for x in values]

        print()
        print(f"CLASS {cls}")

        print(
            f"Voxels > -900 HU : "
            f"{sum(a)/len(a):.2f}%"
        )

        print(
            f"Voxels > -500 HU : "
            f"{sum(b)/len(b):.2f}%"
        )

        print(
            f"Voxels > 0 HU    : "
            f"{sum(c)/len(c):.2f}%"
        )
