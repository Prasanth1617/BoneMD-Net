import torch
import torch.nn as nn
from models.xray_encoder import XRayEncoder

class LateralOnlyStudent(nn.Module):
    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()
        self.xray_encoder = XRayEncoder(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, lateral):
        lateral_features = self.xray_encoder(lateral)
        logits = self.classifier(lateral_features)

        return {
            "logits": logits,
            "lateral_features": lateral_features,
        }
