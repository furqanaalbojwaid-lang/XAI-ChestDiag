"""Step 3 - Evaluate a trained model on the held-out patient-level TEST split.

Writes (in outputs/<run>/seed<seed>/):
    test_predictions.csv / test_metrics.json            plain inference, threshold 0.5
    test_predictions_tta.csv / test_metrics_tta.json    with --tta (mean of original + h-flip)
Each metrics file also contains metrics at the Youden-optimal threshold, which is
chosen on the VALIDATION split only (the test set is never used for any choice).

Usage:
    python -m src.evaluate --model densenet121 --seed 42 [--tta]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from .common import get_device, index_images, load_config
from .dataset import CXRDataset
from .models import count_parameters
from .proposed import load_trained
from .stats import binary_metrics, bootstrap_ci, delong_auc_ci


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit = model(batch[0].to(device))
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(probs)


def predict_df(model, df, index, size, norm, bs, nw, device, tta):
    if device.type == "cuda":
        from .dataset import gpu_preprocess, preload_images
        arr = torch.from_numpy(preload_images([index[i] for i in df.image], size, max(8, 2 * nw)))
        out = []
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            for k in range(0, len(arr), bs):
                xb = arr[k:k + bs].to(device)
                p = torch.sigmoid(model(gpu_preprocess(xb, norm)).float())
                if tta:
                    p = (p + torch.sigmoid(model(gpu_preprocess(xb, norm, hflip_all=True)).float())) / 2
                out.append(p.cpu().numpy())
        return np.concatenate(out)
    dl = DataLoader(CXRDataset(df, index, size, norm), batch_size=bs, num_workers=nw, shuffle=False)
    p = predict(model, dl, device)
    if tta:
        dlf = DataLoader(CXRDataset(df, index, size, norm, hflip=True), batch_size=bs, num_workers=nw, shuffle=False)
        p = (p + predict(model, dlf, device)) / 2
    return p


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="densenet121", help="run name (= output folder)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pc, ec, tc = cfg["preprocessing"], cfg["evaluation"], cfg["training"]
    device = get_device()
    run_dir = f"{cfg['output_dir']}/{args.model}/seed{args.seed}"
    model, _, size = load_trained(run_dir, args.model, device, tc["dropout"])
    size = size or pc["image_size"]
    index = index_images(cfg["data"]["image_root"])
    bs, nw, norm = tc["batch_size"], tc["num_workers"], pc["imagenet_normalize"]

    val_df = pd.read_csv(f"{cfg['data']['split_dir']}/val.csv")
    test_df = pd.read_csv(f"{cfg['data']['split_dir']}/test.csv")
    pv = predict_df(model, val_df, index, size, norm, bs, nw, device, args.tta)
    thr_y = youden_threshold(val_df["label"].values, pv)
    prob = predict_df(model, test_df, index, size, norm, bs, nw, device, args.tta)

    suffix = "_tta" if args.tta else ""
    out = test_df[["image", "patient_id", "findings", "label"]].copy()
    out["prob"] = prob
    out.to_csv(f"{run_dir}/test_predictions{suffix}.csv", index=False)

    y = out["label"].values
    m = binary_metrics(y, prob, ec["threshold"])
    auc, (lo, hi), se = delong_auc_ci(y, prob)
    m["auc_ci95_delong"] = [lo, hi]
    m["ci95_bootstrap"] = bootstrap_ci(y, prob, ec["threshold"], ec["bootstrap_iters"], seed=args.seed)
    m["youden"] = {**binary_metrics(y, prob, thr_y),
                   "ci95_bootstrap": bootstrap_ci(y, prob, thr_y, ec["bootstrap_iters"], seed=args.seed),
                   "threshold_chosen_on": "validation"}
    m["tta"] = bool(args.tta)
    m["image_size"] = size
    m["n_test_images"] = int(len(y))
    m["n_test_abnormal"] = int(y.sum())
    m["n_test_normal"] = int((y == 0).sum())
    m["parameters"] = count_parameters(model)
    with open(f"{run_dir}/test_metrics{suffix}.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items() if k not in ("ci95_bootstrap",)}, indent=2))


if __name__ == "__main__":
    main()
