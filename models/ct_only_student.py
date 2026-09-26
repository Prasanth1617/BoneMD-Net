import torch
import torch.nn as nn

from models.ct_encoder import CTEncoder


class CTOnlyStudent(nn.Module):
    """
    CT-only ablation model.

    Input:
        CT -> [B, 1, 192, 320, 320]

    Output:
        logits -> [B, 3]
    """

    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.ct_encoder = CTEncoder(
            feature_dim=feature_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, ct):

        ct_features = self.ct_encoder(ct)

        logits = self.classifier(
            ct_features
        )

        return {
            "logits": logits,
            "ct_features": ct_features,
        }
