import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_student import BoneMDStudent
from models.bone_md_teacher import BoneMDTeacher
from training.distillation_loss import KnowledgeDistillationLoss


SEED = 42
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

KD_ALPHA = 0.5
FEATURE_BETA = 0.2
KD_TEMPERATURE = 4.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CHECKPOINT = "checkpoints/teacher_v2_controlled_best.pth"
STUDENT_CHECKPOINT = "checkpoints/student_kd_v2_best.pth"


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate(student, teacher, loader, criterion):
    student.eval()
    teacher.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            teacher_output = teacher(ap, lateral, ct)
            student_output = student(ap, lateral, ct)

            losses = criterion(
                student_logits=student_output["logits"],
                teacher_logits=teacher_output["logits"],
                student_features=student_output["fused_features"],
                teacher_features=teacher_output["fused_features"],
                labels=labels,
            )

            running_loss += losses["loss"].item()

            predictions = student_output["logits"].argmax(dim=1)

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("KD alpha:", KD_ALPHA)
    print("Feature beta:", FEATURE_BETA)
    print("Temperature:", KD_TEMPERATURE)
    print("Teacher checkpoint:", TEACHER_CHECKPOINT)
    print("Student checkpoint:", STUDENT_CHECKPOINT)
    print()

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    train_dataset = CachedMultimodalDataset("cache/train")
    val_dataset = CachedMultimodalDataset("cache/val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print()

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

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print(
        "Teacher loaded from epoch:",
        checkpoint["epoch"],
    )

    print(
        "Teacher validation accuracy:",
        checkpoint["val_accuracy"],
    )

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
        alpha=KD_ALPHA,
        beta=FEATURE_BETA,
        temperature=KD_TEMPERATURE,
    ).to(DEVICE)

    # The feature projection inside the KD loss is trainable.
    optimizer = torch.optim.AdamW(
        list(student.parameters()) +
        list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0
    best_epoch = 0

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):

        student.train()

        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_kd_loss = 0.0
        running_feature_loss = 0.0

        correct = 0
        total = 0

        for batch in train_loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Teacher is completely frozen.
            with torch.no_grad():
                teacher_output = teacher(
                    ap,
                    lateral,
                    ct,
                )

            student_output = student(
                ap,
                lateral,
                ct,
            )

            losses = criterion(
                student_logits=student_output["logits"],
                teacher_logits=teacher_output["logits"],
                student_features=student_output["fused_features"],
                teacher_features=teacher_output["fused_features"],
                labels=labels,
            )

            losses["loss"].backward()

            optimizer.step()

            running_total_loss += losses["loss"].detach().item()
            running_cls_loss += losses["classification_loss"].detach().item()
            running_kd_loss += losses["distillation_loss"].detach().item()
            running_feature_loss += losses["feature_loss"].detach().item()

            predictions = student_output["logits"].argmax(dim=1)

            correct += int(
                (predictions == labels).sum().item()
            )

            total += labels.size(0)

        train_loss = running_total_loss / total
        train_cls_loss = running_cls_loss / total
        train_kd_loss = running_kd_loss / total
        train_feature_loss = running_feature_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            student,
            teacher,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        print(
            f"    Components: "
            f"CE={train_cls_loss:.4f} | "
            f"KD={train_kd_loss:.4f} | "
            f"Feature={train_feature_loss:.4f}"
        )

        # --------------------------------------------------------
        # Save best student
        # --------------------------------------------------------
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "kd_loss_state_dict": criterion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "kd_alpha": KD_ALPHA,
                    "feature_beta": FEATURE_BETA,
                    "temperature": KD_TEMPERATURE,
                    "teacher_checkpoint": TEACHER_CHECKPOINT,
                },
                STUDENT_CHECKPOINT,
            )

            print(
                f"    New best student checkpoint saved "
                f"(epoch {epoch})"
            )

    print()
    print("=" * 70)
    print("STUDENT KD TRAINING COMPLETED")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", best_val_accuracy)
    print("Student checkpoint:", STUDENT_CHECKPOINT)


if __name__ == "__main__":
    main()