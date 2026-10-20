"""Proposed XAI-ChestDiag network.

DenseNet121 backbone, optionally extended with
  * CBAM attention (Woo et al., ECCV 2018) on the last dense-block feature maps, and
  * an auxiliary multi-label head that predicts the 14 NIH findings (multi-task learning).

forward(x) returns ONLY the binary triage logit, so evaluation, Grad-CAM and the
Pointing Game work unchanged; forward_all(x) returns (binary_logit, aux_logits).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

LABELS14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def multilabel_targets(findings) -> np.ndarray:
    """'Effusion|Mass' -> 14-dim 0/1 vector (No Finding -> all zeros)."""
    out = np.zeros((len(findings), len(LABELS14)), dtype=np.float32)
    for i, f in enumerate(findings):
        for lab in str(f).split("|"):
            lab = lab.strip()
            if lab in LABELS14:
                out[i, LABELS14.index(lab)] = 1.0
    return out


class ChannelAttention(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(c // r, c, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(c, r)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        return self.sa(self.ca(x))


class XAIChestNet(nn.Module):
    def __init__(self, pretrained: bool = True, attention: str = "none", multitask: bool = False,
                 dropout: float = 0.3, n_aux: int = len(LABELS14)):
        super().__init__()
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features            # ends with norm5 (BatchNorm) -> 1024 x H/32 x W/32
        c = m.classifier.in_features
        self.attn = CBAM(c) if attention == "cbam" else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c, 1)
        self.aux = nn.Linear(c, n_aux) if multitask else None
        self.attention, self.multitask = attention, multitask

    def embed(self, x):
        f = self.attn(F.relu(self.features(x)))
        return torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)

    def forward_all(self, x):
        z = self.dropout(self.embed(x))
        return self.fc(z).squeeze(1), (self.aux(z) if self.aux is not None else None)

    def forward(self, x):
        return self.forward_all(x)[0]


# ------------------------------------------------------------------ loading helpers
def build_from_run_config(rc: dict, pretrained: bool = False) -> nn.Module:
    return XAIChestNet(pretrained=pretrained, attention=rc.get("attention", "none"),
                       multitask=rc.get("multitask", False), dropout=rc.get("dropout", 0.3))


def load_trained(run_dir: str, name: str, device, dropout: float = 0.3):
    """Load best.pt of a run. Proposed runs carry run_config.json; baselines do not.
    Returns (model, gradcam_layer, image_size or None)."""
    from .models import build_model, gradcam_target_layer
    rc_file = Path(run_dir) / "run_config.json"
    if rc_file.exists():
        rc = json.load(open(rc_file))
        model = build_from_run_config(rc).to(device)
        layer = model.attn if rc.get("attention") == "cbam" else model.features.denseblock4
        size = rc.get("image_size")
    else:
        model = build_model(name, pretrained=False, dropout=dropout).to(device)
        layer = gradcam_target_layer(model, name)
        size = None
    model.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=device))
    model.eval()
    return model, layer, size
