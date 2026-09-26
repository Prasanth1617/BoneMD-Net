import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import Dataset


class CachedMultimodalDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("patient_*.pt"))

        if not self.files:
            raise RuntimeError(f"No cached patient files found in {self.cache_dir}")

        self.augment = augment

    def _augment_xray(self, x):
        x = x.squeeze(0).numpy()

        from scipy.ndimage import rotate, shift

        angle = np.random.uniform(-10, 10)
        x = rotate(x, angle, reshape=False, order=1, mode="nearest")

        shift_y, shift_x = np.random.uniform(-8, 8, size=2)
        x = shift(x, [shift_y, shift_x], order=1, mode="nearest")

        if np.random.random() > 0.5:
            x = x[:, ::-1].copy()

        brightness = np.random.uniform(0.85, 1.15)
        x = x * brightness

        x = np.clip(x, 0.0, 1.0).astype(np.float32)

        return torch.from_numpy(x).unsqueeze(0)

    def _augment_ct(self, ct):
        ct = ct.squeeze(0).numpy()

        from scipy.ndimage import rotate, shift

        angle = np.random.uniform(-3, 3)
        ct = rotate(ct, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

        shifts = np.random.uniform(-3, 3, size=3)
        ct = shift(ct, shifts, order=1, mode="nearest")

        ct = np.clip(ct, -1000, 1000).astype(np.float32)

        return torch.from_numpy(ct).unsqueeze(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        sample = torch.load(path, map_location="cpu")

        ap = sample["ap"]
        lateral = sample["lateral"]
        ct = sample["ct"]

        if self.augment:
            ap = self._augment_xray(ap)
            lateral = self._augment_xray(lateral)
            ct = self._augment_ct(ct)

        return {
            "ap": ap,
            "lateral": lateral,
            "ct": ct,
            "label": sample["label"],
            "patient_id": sample["patient_id"],
            "split": sample["split"],
        }
