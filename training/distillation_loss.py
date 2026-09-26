import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined classification + knowledge-distillation loss.

    Components:
        1. Hard-label cross-entropy
        2. Soft-logit distillation
        3. Feature distillation

    The teacher is assumed to be frozen during student training.
    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.2,
        temperature=4.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.feature_projection = nn.Linear(256, 512)

    def forward(
        self,
        student_logits,
        teacher_logits,
        student_features,
        teacher_features,
        labels,
    ):
        classification_loss = F.cross_entropy(
            student_logits,
            labels,
        )

        temperature = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / temperature,
            dim=1,
        )

        teacher_probs = F.softmax(
            teacher_logits / temperature,
            dim=1,
        )

        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)

        projected_student = self.feature_projection(
            student_features
        )

        feature_loss = F.mse_loss(
            projected_student,
            teacher_features.detach(),
        )

        total_loss = (
            (1.0 - self.alpha - self.beta) * classification_loss
            + self.alpha * distillation_loss
            + self.beta * feature_loss
        )

        return {
            "loss": total_loss,
            "classification_loss": classification_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
        }
