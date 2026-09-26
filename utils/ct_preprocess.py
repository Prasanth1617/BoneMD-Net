import numpy as np
from scipy.ndimage import zoom

TARGET_Z = 192
TARGET_H = 320
TARGET_W = 320

TARGET_Z_SPACING = 2.5
TARGET_INPLANE_SPACING = 1.0
PADDING_HU = -1024.0


def resample_ct(volume, metadata):
    """
    Resample a CT volume to:
        in-plane spacing: 1.0 x 1.0 mm
        Z spacing:         2.5 mm
    """
    volume = volume.astype(np.float32)

    z_spacing = abs(
        float(metadata[-1]["z"]) - float(metadata[0]["z"])
    ) / (len(metadata) - 1)

    row_spacing = float(metadata[0]["row_spacing"])
    col_spacing = float(metadata[0]["col_spacing"])

    zoom_factors = (
        z_spacing / TARGET_Z_SPACING,
        row_spacing / TARGET_INPLANE_SPACING,
        col_spacing / TARGET_INPLANE_SPACING,
    )

    return zoom(volume, zoom_factors, order=1).astype(np.float32)


def center_crop_or_pad(volume):
    """
    Convert the volume to exactly:
        [192, 320, 320]

    Padding uses -1024 HU.
    """
    volume = volume.astype(np.float32)

    output = np.full(
        (TARGET_Z, TARGET_H, TARGET_W),
        PADDING_HU,
        dtype=np.float32,
    )

    z, h, w = volume.shape

    # Center crop/pad in Z.
    if z >= TARGET_Z:
        z_start = (z - TARGET_Z) // 2
        volume_z = volume[z_start:z_start + TARGET_Z]
        out_z_start = 0
    else:
        volume_z = volume
        out_z_start = (TARGET_Z - z) // 2

    # Center crop/pad in height.
    if h >= TARGET_H:
        h_start = (h - TARGET_H) // 2
        volume_z = volume_z[:, h_start:h_start + TARGET_H]
        out_h_start = 0
    else:
        out_h_start = (TARGET_H - h) // 2

    # Center crop/pad in width.
    if w >= TARGET_W:
        w_start = (w - TARGET_W) // 2
        volume_z = volume_z[:, :, w_start:w_start + TARGET_W]
        out_w_start = 0
    else:
        out_w_start = (TARGET_W - w) // 2

    out_z_end = out_z_start + volume_z.shape[0]
    out_h_end = out_h_start + volume_z.shape[1]
    out_w_end = out_w_start + volume_z.shape[2]

    output[
        out_z_start:out_z_end,
        out_h_start:out_h_end,
        out_w_start:out_w_end,
    ] = volume_z

    return output


def preprocess_ct(volume, metadata):
    """
    Complete fixed-size CT preprocessing.

    Returns:
        float32 array with shape [192, 320, 320]
    """
    volume = resample_ct(volume, metadata)
    volume = center_crop_or_pad(volume)

    return np.nan_to_num(
        volume,
        nan=PADDING_HU,
        posinf=4000.0,
        neginf=PADDING_HU,
    ).astype(np.float32)
