import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_student import BoneMDStudent
from models.bone_md_teacher import BoneMDTeacher
from training.distillation_loss import KnowledgeDistillationLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CHECKPOINT = "checkpoints/teacher_v2_controlled_best.pth"


def main():
    print("Device:", DEVICE)
    print("Teacher checkpoint:", TEACHER_CHECKPOINT)
    print()

    dataset = CachedMultimodalDataset("cache/train")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    ap = batch["ap"].to(DEVICE)
    lateral = batch["lateral"].to(DEVICE)
    ct = batch["ct"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    # ------------------------------------------------------------
    # Frozen teacher
    # ------------------------------------------------------------
    teacher = BoneMDTeacher(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        TEACHER_CHECKPOINT,
        map_location=DEVICE,
    )

    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # ------------------------------------------------------------
    # Student
    # ------------------------------------------------------------
    student = BoneMDStudent(
        feature_dim=256,
        num_classes=3,
    ).to(DEVICE)

    # ------------------------------------------------------------
    # KD loss
    # ------------------------------------------------------------
    criterion = KnowledgeDistillationLoss(
        alpha=0.5,
        beta=0.2,
        temperature=4.0,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------
    # Teacher forward — completely frozen
    # ------------------------------------------------------------
    with torch.no_grad():
        teacher_output = teacher(
            ap,
            lateral,
            ct,
        )

    print("Teacher output keys:", teacher_output.keys())

    # ------------------------------------------------------------
    # Student forward
    # ------------------------------------------------------------
    student.train()

    optimizer.zero_grad(set_to_none=True)

    student_output = student(
        ap,
        lateral,
        ct,
    )

    print("Student output keys:", student_output.keys())

    # ------------------------------------------------------------
    # KD loss
    # ------------------------------------------------------------
    losses = criterion(
        student_logits=student_output["logits"],
        teacher_logits=teacher_output["logits"],
        student_features=student_output["fused_features"],
        teacher_features=teacher_output["fused_features"],
        labels=labels,
    )

    losses["loss"].backward()
    optimizer.step()

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------
    print("Batch size:", labels.size(0))
    print("Patient ID:", batch["patient_id"])
    print("Label:", labels.cpu().tolist())

    print()
    print("Student logits shape:", tuple(student_output["logits"].shape))
    print("Teacher logits shape:", tuple(teacher_output["logits"].shape))

    print()
    print("Total loss:", float(losses["loss"].detach().cpu()))
    print("Classification loss:", float(losses["classification_loss"].detach().cpu()))
    print("Distillation loss:", float(losses["distillation_loss"].detach().cpu()))
    print("Feature loss:", float(losses["feature_loss"].detach().cpu()))

    print()
    print(
        "Student logits finite:",
        bool(torch.isfinite(student_output["logits"]).all()),
    )

    print(
        "Teacher logits finite:",
        bool(torch.isfinite(teacher_output["logits"]).all()),
    )

    print(
        "Loss finite:",
        bool(torch.isfinite(losses["loss"])),
    )

    student_gradients_finite = all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all()
        for parameter in student.parameters()
    )

    teacher_gradients_present = any(
        parameter.grad is not None
        for parameter in teacher.parameters()
    )

    print(
        "Student gradients finite:",
        bool(student_gradients_finite),
    )

    print(
        "Teacher gradients present:",
        bool(teacher_gradients_present),
    )

    print()
    print("=" * 70)
    print("KD ONE-BATCH TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()