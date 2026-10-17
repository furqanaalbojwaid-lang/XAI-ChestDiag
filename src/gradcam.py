"""Grad-CAM (Selvaraju et al., ICCV 2017).

    alpha_k^c = 1/Z * sum_i sum_j dY^c / dA_ij^k
    L^c       = ReLU( sum_k alpha_k^c A^k )

The heat-map is up-sampled to the input size and min-max normalised to [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        # in-place ReLUs would overwrite the stored activations / break hooks
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.acts = None
        self.grads = None
        self._h = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        self.acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, g):
        self.grads = g

    def remove(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, target: str = "abnormal") -> tuple[np.ndarray, np.ndarray]:
        """x: (B,3,H,W). target='abnormal' explains the positive logit,
        'normal' explains the negative logit. Returns (cams [B,H,W] in [0,1], probs [B])."""
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        score = logit if target == "abnormal" else -logit
        score.sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # alpha_k^c
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # ReLU(sum_k alpha A^k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach()
        b = cam.shape[0]
        flat = cam.view(b, -1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)
        return cam.cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()
