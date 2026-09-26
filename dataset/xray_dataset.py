import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.xray_loader import load_xray
from utils.xray_preprocess import preprocess_xray


class LUMOSXrayDataset(Dataset):
    """
    Patient-level LUMOS X-ray dataset.

    Each sample contains:
        AP X-ray       -> [1, 224, 224]
        Lateral X-ray -> [1, 224, 224]
        label          -> 0, 1, or 2
        patient_id
    """

    def __init__(
        self,
        manifest_path,
        xray_zip_path,
    ):
        self.manifest = pd.read_csv(manifest_path)
        self.xray_zip_path = xray_zip_path

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        ap_image, _ = load_xray(
            self.xray_zip_path,
            row["ap_dicom_file"],
        )

        lateral_image, _ = load_xray(
            self.xray_zip_path,
            row["lateral_dicom_file"],
        )

        ap_image = preprocess_xray(ap_image)
        lateral_image = preprocess_xray(lateral_image)

        ap_image = torch.from_numpy(ap_image).unsqueeze(0)
        lateral_image = torch.from_numpy(lateral_image).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return {
            "ap": ap_image,
            "lateral": lateral_image,
            "label": label,
            "patient_id": int(row["patient_id"]),
        }
