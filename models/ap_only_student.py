import torch
import torch.nn as nn

from models.xray_encoder import XRayEncoder


class APOnlyStudent(nn.Module):
    """
    AP-only ablation model.

    Input:
        AP X-ray -> [B, 1, 224, 224]

    Output:
        logits -> [B, 3]
    """

    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.xray_encoder = XRayEncoder(
            feature_dim=feature_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, ap):

        ap_features = self.xray_encoder(ap)

        logits = self.classifier(
            ap_features
        )

        return {
            "logits": logits,
            "ap_features": ap_features,
        }
