"""Step 8 - Grad-CAM qualitative figure from REAL test-set predictions.

Automatically selects, from the held-out test split, the most confidently
and correctly classified example for each requested finding plus one
correctly classified Normal case, and draws [radiograph | Grad-CAM overlay]
pairs. A CSV lists the exact images used, so the figure is reproducible.

Usage:
    python -m src.gradcam_figure --model densenet121 --seed 42 \
        --findings Pneumonia Infiltration Effusion
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import ensure_dir, get_device, index_images, load_config
from .dataset import preprocess
from .gradcam import GradCAM
from .models import build_model, gradcam_target_layer


def overlay(gray: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    base = np.repeat(gray[..., None] / 255.0, 3, axis=2)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def pick_cases(pred: pd.DataFrame, findings: list[str], thr: float) -> list[tuple[str, str]]:
    chosen, used = [], set()
    for f in findings:
        cand = pred[(pred.label == 1) & (pred.prob >= thr) & pred.findings.str.contains(f, regex=False)]
        cand = cand[~cand.image.isin(used)].sort_values("prob", ascending=False)
        if len(cand):
            r = cand.iloc[0]
            chosen.append((r.image, f))
            used.add(r.image)
        else:
            print(f"[warn] no correctly classified test image with finding '{f}'")
    normal = pred[(pred.label == 0) & (pred.prob < thr)].sort_values("prob")
    if len(normal):
        chosen.append((normal.iloc[0].image, "No Finding"))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--findings", nargs="+", default=["Pneumonia", "Infiltration", "Effusion"])
    ap.add_argument("--images", nargs="*", help="explicit image names instead of automatic selection")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pc, tc, ec = cfg["preprocessing"], cfg["training"], cfg["evaluation"]
    device = get_device()
    run_dir = f"{cfg['output_dir']}/{args.model}/seed{args.seed}"
    pred = pd.read_csv(f"{run_dir}/test_predictions.csv")

    if args.images:
        cases = [(im, pred.set_index("image").loc[im, "findings"]) for im in args.images]
    else:
        cases = pick_cases(pred, args.findings, ec["threshold"])

    model = build_model(args.model, pretrained=False, dropout=tc["dropout"]).to(device)
    model.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=device))
    engine = GradCAM(model, gradcam_target_layer(model, args.model))
    index = index_images(cfg["data"]["image_root"])

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.7 * n))
    axes = np.atleast_2d(axes)
    records = []
    for i, (image, label) in enumerate(cases):
        x, gray = preprocess(index[image], pc["image_size"], pc["imagenet_normalize"])
        cam, prob = engine(x.unsqueeze(0).to(device), target="abnormal")
        axes[i, 0].imshow(gray, cmap="gray")
        im = axes[i, 1].imshow(overlay(gray, cam[0]))
        axes[i, 0].set_title(f"({chr(97 + i)}) {label}", fontsize=9, loc="left")
        axes[i, 1].set_title(f"Grad-CAM, P(abnormal) = {prob[0]:.3f}", fontsize=9, loc="left")
        for ax in axes[i]:
            ax.axis("off")
        records.append(dict(panel=chr(97 + i), image=image, findings=label, prob_abnormal=float(prob[0])))
    engine.remove()
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes[:, 1].tolist(), fraction=0.04, pad=0.02)
    cbar.set_label("Grad-CAM activation (normalised)")

    out = ensure_dir(f"{cfg['output_dir']}/figures")
    fig.savefig(out / "fig_gradcam_cases.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_gradcam_cases.pdf", bbox_inches="tight")
    pd.DataFrame(records).to_csv(out / "fig_gradcam_cases_images.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
