import torch
import torch.nn as nn


class TeacherXRayEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class TeacherCTEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.projection = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class BoneMDTeacherTScore(nn.Module):
    """
    Multimodal teacher with two training objectives:

    1. Three-class bone-disorder classification.
    2. Auxiliary prediction of six lumbar T-scores.

    The T-score head is used only during training.
    It is not required during inference.
    """

    def __init__(self, feature_dim=512, num_classes=3):
        super().__init__()

        self.ap_encoder = TeacherXRayEncoder(feature_dim)
        self.lateral_encoder = TeacherXRayEncoder(feature_dim)
        self.ct_encoder = TeacherCTEncoder(feature_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Six lumbar T-score regression targets:
        # L1-L2, L1-L3, L1-L4, L2-L3, L2-L4, L3-L4
        self.tscore_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, ap, lateral, ct):
        ap_features = self.ap_encoder(ap)
        lateral_features = self.lateral_encoder(lateral)
        ct_features = self.ct_encoder(ct)

        combined = torch.cat(
            [ap_features, lateral_features, ct_features],
            dim=1,
        )

        fused_features = self.fusion(combined)

        logits = self.classifier(fused_features)

        tscore_predictions = self.tscore_head(fused_features)

        return {
            "logits": logits,
            "tscore_predictions": tscore_predictions,
            "fused_features": fused_features,
            "ap_features": ap_features,
            "lateral_features": lateral_features,
            "ct_features": ct_features,
        }