"""Step 7 - Publication figures (300 dpi PNG + vector PDF), all from real outputs.

    fig_confusion_matrix   - proposed model, test set
    fig_roc_curves         - all models on the same test set, with AUCs
    fig_training_curves    - train / val loss and accuracy of the proposed model

Usage:
    python -m src.plots --seed 42 --proposed densenet121
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .common import ensure_dir, load_config

NICE = {"densenet121": "DenseNet121 (proposed)", "vgg16": "VGG16",
        "inception_v3": "InceptionV3", "efficientnet_b4": "EfficientNet-B4"}
COLORS = {"densenet121": "#C0392B", "vgg16": "#7F8C8D", "inception_v3": "#2E86C1", "efficientnet_b4": "#28B463"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", path.with_suffix(".png"))


def confusion_matrix_fig(m: dict, path: Path):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Normal", "Abnormal"])
    ax.set_yticks([0, 1], ["Normal", "Abnormal"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.spines[:].set_visible(False)
    ax.set_title(f"Test set (N = {cm.sum():,})", fontsize=11)
    save(fig, path)


def roc_fig(preds: dict[str, pd.DataFrame], metrics: dict[str, dict], path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(p.label, p.prob)
        lo, hi = metrics[name]["auc_ci95_delong"]
        ax.plot(fpr, tpr, lw=2 if name == "densenet121" else 1.4, color=COLORS.get(name),
                label=f"{NICE.get(name, name)}: AUC {metrics[name]['auc']:.3f} ({lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    save(fig, path)


def training_fig(hist: pd.DataFrame, best_epoch: int, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in [(axes[0], "loss", "Binary cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(hist.epoch, hist[f"train_{key}"], label="Training", color="#2E86C1", lw=1.8)
        ax.plot(hist.epoch, hist[f"val_{key}"], label="Validation", color="#C0392B", lw=1.8)
        ax.axvline(best_epoch, ls=":", color="#555", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        ax.legend(frameon=False)
    axes[0].text(best_epoch, axes[0].get_ylim()[1], f" best epoch = {best_epoch}", va="top", fontsize=9)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proposed", default="densenet121")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    fig_dir = ensure_dir(root / "figures")

    preds, metrics = {}, {}
    for name in cfg["models"]:
        d = root / name / f"seed{args.seed}"
        if (d / "test_predictions.csv").exists():
            preds[name] = pd.read_csv(d / "test_predictions.csv")
            metrics[name] = json.load(open(d / "test_metrics.json"))
    if args.proposed in metrics:
        confusion_matrix_fig(metrics[args.proposed], fig_dir / "fig_confusion_matrix")
    if preds:
        roc_fig(preds, metrics, fig_dir / "fig_roc_curves")
    d = root / args.proposed / f"seed{args.seed}"
    if (d / "history.csv").exists():
        info = json.load(open(d / "train_info.json"))
        training_fig(pd.read_csv(d / "history.csv"), info["best_epoch"], fig_dir / "fig_training_curves")


if __name__ == "__main__":
    main()
