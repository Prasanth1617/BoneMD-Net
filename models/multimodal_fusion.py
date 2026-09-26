import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """
    Fuses AP X-ray, lateral X-ray, and CT feature vectors.

    Inputs:
        ap       -> [B, 256]
        lateral  -> [B, 256]
        ct       -> [B, 256]

    Output:
        fused    -> [B, 256]
    """

    def __init__(self, feature_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, ap, lateral, ct):
        combined = torch.cat([ap, lateral, ct], dim=1)

        weights = torch.softmax(self.attention(combined), dim=1)

        ap = ap * weights[:, 0:1]
        lateral = lateral * weights[:, 1:2]
        ct = ct * weights[:, 2:3]

        weighted = torch.cat([ap, lateral, ct], dim=1)

        fused = self.projection(weighted)

        return fused, weights
