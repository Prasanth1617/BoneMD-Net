import torch
import torch.nn as nn

from models.xray_encoder import XRayEncoder


class XRayOnlyStudent(nn.Module):
    """
    X-ray-only ablation model.

    Inputs:
        AP X-ray       -> [B, 1, 224, 224]
        Lateral X-ray  -> [B, 1, 224, 224]

    Output:
        logits         -> [B, 3]
    """

    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.xray_encoder = XRayEncoder(
            feature_dim=feature_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, ap, lateral):
        ap_features = self.xray_encoder(ap)
        lateral_features = self.xray_encoder(lateral)

        combined_features = torch.cat(
            [ap_features, lateral_features],
            dim=1,
        )

        logits = self.classifier(
            combined_features
        )

        return {
            "logits": logits,
            "ap_features": ap_features,
            "lateral_features": lateral_features,
        }
