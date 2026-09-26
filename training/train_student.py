import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.multimodal_dataset import LUMOSMultimodalDataset
from models.bone_md_student import BoneMDStudent
from models.bone_md_teacher import BoneMDTeacher
from training.config import (
    BATCH_SIZE,
    CT_BASE_DIR,
    DEVICE,
    FEATURE_BETA,
    KD_ALPHA,
    KD_TEMPERATURE,
    LEARNING_RATE,
    MANIFEST_PATH,
    NUM_WORKERS,
    NUM_CLASSES,
    XRAY_ZIP_PATH,
)
from training.distillation_loss import KnowledgeDistillationLoss


RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(RANDOM_SEED)

    print("Device:", DEVICE)

    train_dataset = LUMOSMultimodalDataset(
        MANIFEST_PATH,
        XRAY_ZIP_PATH,
        CT_BASE_DIR,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    student = BoneMDStudent(
        feature_dim=256,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    teacher = BoneMDTeacher(
        feature_dim=512,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    loss_fn = KnowledgeDistillationLoss(
        alpha=KD_ALPHA,
        beta=FEATURE_BETA,
        temperature=KD_TEMPERATURE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    batch = next(iter(train_loader))

    ap = batch["ap"].to(DEVICE, non_blocking=True)
    lateral = batch["lateral"].to(DEVICE, non_blocking=True)
    ct = batch["ct"].to(DEVICE, non_blocking=True)
    labels = batch["label"].to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        teacher_output = teacher(ap, lateral, ct)

    student_output = student(ap, lateral, ct)

    losses = loss_fn(
        student_logits=student_output["logits"],
        teacher_logits=teacher_output["logits"],
        student_features=student_output["fused_features"],
        teacher_features=teacher_output["fused_features"],
        labels=labels,
    )

    losses["loss"].backward()

    optimizer.step()

    print("One training batch completed successfully")
    print("Batch size:", ap.shape[0])
    print("Labels:", labels.detach().cpu().tolist())
    print("Student logits:", student_output["logits"].shape)
    print("Teacher logits:", teacher_output["logits"].shape)
    print("Total loss:", losses["loss"].item())
    print("Classification loss:", losses["classification_loss"].item())
    print("Distillation loss:", losses["distillation_loss"].item())
    print("Feature loss:", losses["feature_loss"].item())

    student_gradients = [
        parameter.grad
        for parameter in student.parameters()
        if parameter.grad is not None
    ]

    print(
        "Student gradients finite:",
        all(torch.isfinite(g).all().item() for g in student_gradients),
    )

    print(
        "Teacher gradients:",
        any(parameter.grad is not None for parameter in teacher.parameters()),
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
