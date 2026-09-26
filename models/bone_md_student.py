import torch
import torch.nn as nn

from models.xray_encoder import XRayEncoder
from models.ct_encoder import CTEncoder
from models.multimodal_fusion import MultimodalFusion
from models.classifier import BoneDisorderClassifier


class BoneMDStudent(nn.Module):
    """
    BoneMD-Net lightweight multimodal student model.

    Inputs:
        AP X-ray     -> [B, 1, 224, 224]
        Lateral X-ray -> [B, 1, 224, 224]
        CT           -> [B, 1, 192, 320, 320]

    Outputs:
        logits       -> [B, 3]
        fusion_weights -> [B, 3]
    """

    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.xray_encoder = XRayEncoder(
            feature_dim=feature_dim
        )

        self.ct_encoder = CTEncoder(
            feature_dim=feature_dim
        )

        self.fusion = MultimodalFusion(
            feature_dim=feature_dim
        )

        self.classifier = BoneDisorderClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes
        )

    def forward(self, ap, lateral, ct):
        ap_features = self.xray_encoder(ap)
        lateral_features = self.xray_encoder(lateral)
        ct_features = self.ct_encoder(ct)

        fused_features, fusion_weights = self.fusion(
            ap_features,
            lateral_features,
            ct_features
        )

        logits = self.classifier(fused_features)

        return {
            "logits": logits,
            "fused_features": fused_features,
            "ap_features": ap_features,
            "lateral_features": lateral_features,
            "ct_features": ct_features,
            "fusion_weights": fusion_weights,
        }
