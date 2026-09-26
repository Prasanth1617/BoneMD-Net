import torch
import torch.nn as nn


class BoneDisorderClassifier(nn.Module):
    """
    Classification head for the fused multimodal representation.

    Input:
        [B, 256]

    Output:
        [B, 3] logits

    Classes:
        0 -> Normal
        1 -> Osteopenia
        2 -> Osteoporosis
    """

    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
