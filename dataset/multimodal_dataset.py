import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.xray_loader import load_xray
from utils.xray_preprocess import preprocess_xray
from utils.ct_loader import load_ct_volume
from utils.ct_preprocess import preprocess_ct


class LUMOSMultimodalDataset(Dataset):
    """
    Patient-level LUMOS multimodal dataset.

    Each sample contains:
        AP X-ray     -> [1, 224, 224]
        Lateral X-ray -> [1, 224, 224]
        CT           -> [1, 192, 320, 320]
        label        -> 0, 1, or 2
        patient_id
        split
    """

    def __init__(self, manifest_path, xray_zip_path, ct_base_dir, split=None):
        self.manifest = pd.read_csv(manifest_path)

        if split is not None:
            self.manifest = self.manifest[
                self.manifest["split"] == split
            ].reset_index(drop=True)

        self.xray_zip_path = xray_zip_path
        self.ct_base_dir = ct_base_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        # X-rays
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

        # CT
        ct_zip_path = f"{self.ct_base_dir}/{row['ct_zip']}"

        ct_volume, ct_metadata = load_ct_volume(
            ct_zip_path,
            row["ct_patient_folder"],
        )

        ct_volume = preprocess_ct(
            ct_volume,
            ct_metadata,
        )

        ct_volume = torch.from_numpy(ct_volume).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return {
            "ap": ap_image,
            "lateral": lateral_image,
            "ct": ct_volume,
            "label": label,
            "patient_id": int(row["patient_id"]),
            "split": row["split"],
        }
