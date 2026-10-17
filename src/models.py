"""Model zoo: DenseNet121 (proposed) and baselines VGG16, InceptionV3, EfficientNet-B4.

Every network ends with the same binary head:
    backbone features -> Global Average Pooling -> Dropout(p) -> Linear(1) -> (Sigmoid in the loss)
so that the comparison is fair. For VGG16 this means the original 3 fully
connected layers (~124 M parameters) are replaced by GAP + Linear, which keeps
the model at ~14.7 M parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BinaryHeadNet(nn.Module):
    def __init__(self, features: nn.Module, n_features: int, dropout: float, final_relu: bool = False):
        super().__init__()
        self.features = features
        self.final_relu = final_relu
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        f = self.features(x)
        if self.final_relu:
            f = F.relu(f)
        f = torch.flatten(self.pool(f), 1)
        return self.fc(self.dropout(f)).squeeze(1)  # logits


def build_model(name: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        # torchvision applies ReLU after features.norm5 inside forward(); we keep that.
        return BinaryHeadNet(m.features, m.classifier.in_features, dropout, final_relu=True)
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        return BinaryHeadNet(m.features, 512, dropout)
    if name == "efficientnet_b4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        return BinaryHeadNet(m.features, m.classifier[1].in_features, dropout)
    if name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
                                aux_logits=True, init_weights=not pretrained)
        m.transform_input = False  # we already normalise with ImageNet statistics
        feats = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, m.maxpool1,
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, m.maxpool2,
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d, m.Mixed_6a, m.Mixed_6b, m.Mixed_6c,
            m.Mixed_6d, m.Mixed_6e, m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
        )
        return BinaryHeadNet(feats, 2048, dropout)
    raise ValueError(f"Unknown model: {name}")


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional feature block used for Grad-CAM."""
    name = name.lower()
    if name == "densenet121":
        return model.features.denseblock4      # output of the final dense block (7x7x1024)
    if name == "vgg16":
        return model.features[28]              # last conv layer
    if name == "efficientnet_b4":
        return model.features[-1]
    if name == "inception_v3":
        return model.features[-1]              # Mixed_7c
    raise ValueError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
