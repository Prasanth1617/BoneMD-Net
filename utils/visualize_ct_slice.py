from pathlib import Path

import matplotlib.pyplot as plt

from ct_loader import load_ct_volume


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = PROJECT_ROOT / "dataset" / "lumos_ct_141_210_dcm.zip"
PATIENT_FOLDER = "lumos_ct_175"


def main():
    volume, metadata = load_ct_volume(
        ZIP_PATH,
        PATIENT_FOLDER,
    )

    middle_index = volume.shape[0] // 2
    image = volume[middle_index]

    # Bone-focused display window
    window_min = -200
    window_max = 1000

    plt.figure(figsize=(7, 7))
    plt.imshow(
        image,
        cmap="gray",
        vmin=window_min,
        vmax=window_max,
    )
    plt.axis("off")
    plt.title(
        f"{PATIENT_FOLDER} | "
        f"slice {middle_index + 1}/{volume.shape[0]}"
    )

    output_path = PROJECT_ROOT / "results" / "ct_visual_check_175.png"
    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    print("CT visualization created successfully.")
    print("Patient:", PATIENT_FOLDER)
    print("Volume shape:", volume.shape)
    print("Displayed slice:", middle_index)
    print("HU range:", float(image.min()), "to", float(image.max()))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()