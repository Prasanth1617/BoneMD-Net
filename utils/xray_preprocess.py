import cv2
import numpy as np


TARGET_SIZE = 224


def percentile_normalize(image, low=1.0, high=99.0):
    """
    Normalize one X-ray image using per-image percentiles.

    Input:
        float32 image with original DICOM intensity values.

    Output:
        float32 image in [0, 1].
    """
    image = image.astype(np.float32)

    p_low, p_high = np.percentile(
        image,
        [low, high],
    )

    if p_high <= p_low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(
        image,
        p_low,
        p_high,
    )

    image = (
        image - p_low
    ) / (
        p_high - p_low
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


def resize_and_pad(image, target_size=TARGET_SIZE):
    """
    Resize while preserving aspect ratio, then zero-pad
    to a square target size.

    Input:
        2D float32 image in [0, 1].

    Output:
        2D float32 image of shape (target_size, target_size).
    """
    height, width = image.shape

    scale = min(
        target_size / height,
        target_size / width,
    )

    new_height = max(
        1,
        round(height * scale),
    )

    new_width = max(
        1,
        round(width * scale),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    output = np.zeros(
        (target_size, target_size),
        dtype=np.float32,
    )

    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2

    output[
        top:top + new_height,
        left:left + new_width,
    ] = resized

    return output


def preprocess_xray(
    image,
    target_size=TARGET_SIZE,
):
    """
    Complete X-ray preprocessing pipeline.

    DICOM intensity image
        -> percentile normalization
        -> aspect-preserving resize
        -> zero padding
        -> fixed-size output
    """
    image = percentile_normalize(image)

    image = resize_and_pad(
        image,
        target_size=target_size,
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)
