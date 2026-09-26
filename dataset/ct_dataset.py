import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.ct_loader import load_ct_volume
from utils.ct_preprocess import preprocess_ct


class LUMOSCTDataset(Dataset):
    """
    Patient-level LUMOS CT dataset.

    Each sample contains:
        CT volume -> [1, 192, 320, 320]
        label     -> 0, 1, or 2
        patient_id
    """

    def __init__(self, manifest_path, ct_base_dir):
        self.manifest = pd.read_csv(manifest_path)
        self.ct_base_dir = ct_base_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        zip_path = f"{self.ct_base_dir}/{row['ct_zip']}"
        patient_folder = row["ct_patient_folder"]

        volume, metadata = load_ct_volume(
            zip_path,
            patient_folder,
        )

        volume = preprocess_ct(volume, metadata)

        volume = torch.from_numpy(volume).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return {
            "ct": volume,
            "label": label,
            "patient_id": int(row["patient_id"]),
        }
