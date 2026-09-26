import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.multimodal_dataset import LUMOSMultimodalDataset
from models.bone_md_teacher import BoneMDTeacher
from training.config import (
    BATCH_SIZE,
    CT_BASE_DIR,
    DEVICE,
    LEARNING_RATE,
    MANIFEST_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    XRAY_ZIP_PATH,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    print("Device:", DEVICE)

    dataset = LUMOSMultimodalDataset(
        MANIFEST_PATH,
        XRAY_ZIP_PATH,
        CT_BASE_DIR,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    teacher = BoneMDTeacher(
        feature_dim=512,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    teacher.train()

    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = torch.nn.CrossEntropyLoss()

    batch = next(iter(loader))

    ap = batch["ap"].to(DEVICE, non_blocking=True)
    lateral = batch["lateral"].to(DEVICE, non_blocking=True)
    ct = batch["ct"].to(DEVICE, non_blocking=True)
    labels = batch["label"].to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    output = teacher(ap, lateral, ct)

    loss = criterion(
        output["logits"],
        labels,
    )

    loss.backward()
    optimizer.step()

    print("One teacher training batch completed successfully")
    print("Batch size:", ap.shape[0])
    print("Labels:", labels.detach().cpu().tolist())
    print("Logits:", output["logits"].shape)
    print("Loss:", loss.item())

    gradients = [
        parameter.grad
        for parameter in teacher.parameters()
        if parameter.grad is not None
    ]

    print(
        "Teacher gradients finite:",
        all(torch.isfinite(g).all().item() for g in gradients),
    )

    if DEVICE.type == "cuda":
        print(
            "Peak VRAM:",
            round(
                torch.cuda.max_memory_allocated(DEVICE) / 1024**2,
                1,
            ),
            "MiB",
        )


if __name__ == "__main__":
    main()

