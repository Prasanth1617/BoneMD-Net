import zipfile
import numpy as np
import pydicom


def load_xray(zip_path, dicom_path):
    """
    Load one LUMOS X-ray directly from the ZIP archive.

    Returns the original pixel values as float32.
    No 8-bit conversion is performed here.
    """

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(dicom_path) as f:
            ds = pydicom.dcmread(f)

    image = ds.pixel_array.astype(np.float32)

    # Standardize grayscale polarity.
    # MONOCHROME1 means lower stored values represent higher luminance.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        bits_stored = int(ds.BitsStored)
        max_value = float((1 << bits_stored) - 1)
        image = max_value - image

    return image, ds


if __name__ == "__main__":
    zip_path = "dataset/lumos_x_001_280_dcm.zip"
    dicom_path = "lumos_x_001/lumos_x_001_2.Dcm"

    image, ds = load_xray(zip_path, dicom_path)

    print("X-ray loaded successfully")
    print("Shape:", image.shape)
    print("dtype:", image.dtype)
    print("Minimum:", float(image.min()))
    print("Maximum:", float(image.max()))
    print("Mean:", float(image.mean()))
    print("Modality:", getattr(ds, "Modality", None))
    print("SeriesDescription:", getattr(ds, "SeriesDescription", None))
    print("BitsAllocated:", getattr(ds, "BitsAllocated", None))
    print("BitsStored:", getattr(ds, "BitsStored", None))
    print("PhotometricInterpretation:", getattr(ds, "PhotometricInterpretation", None))
