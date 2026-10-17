"""Step 3 - Evaluate a trained model on the held-out patient-level TEST split.

Writes:
    outputs/<model>/seed<seed>/test_predictions.csv   (one row per test image)
    outputs/<model>/seed<seed>/test_metrics.json      (metrics + 95% CIs)

Usage:
    python -m src.evaluate --model densenet121 --seed 42
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import get_device, index_images, load_config
from .dataset import CXRDataset
from .models import build_model, count_parameters
from .stats import binary_metrics, bootstrap_ci, delong_auc_ci


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pc, ec, tc = cfg["preprocessing"], cfg["evaluation"], cfg["training"]
    device = get_device()
    run_dir = f"{cfg['output_dir']}/{args.model}/seed{args.seed}"

    test_df = pd.read_csv(f"{cfg['data']['split_dir']}/test.csv")
    index = index_images(cfg["data"]["image_root"])
    ds = CXRDataset(test_df, index, pc["image_size"], pc["imagenet_normalize"])
    dl = DataLoader(ds, batch_size=tc["batch_size"], num_workers=tc["num_workers"], shuffle=False)

    model = build_model(args.model, pretrained=False, dropout=tc["dropout"]).to(device)
    model.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=device))
    prob = predict(model, dl, device)

    out = test_df[["image", "patient_id", "findings", "label"]].copy()
    out["prob"] = prob
    out.to_csv(f"{run_dir}/test_predictions.csv", index=False)

    y = out["label"].values
    m = binary_metrics(y, prob, ec["threshold"])
    auc, (lo, hi), se = delong_auc_ci(y, prob)
    m["auc_ci95_delong"] = [lo, hi]
    m["ci95_bootstrap"] = bootstrap_ci(y, prob, ec["threshold"], ec["bootstrap_iters"], seed=args.seed)
    m["n_test_images"] = int(len(y))
    m["n_test_abnormal"] = int(y.sum())
    m["n_test_normal"] = int((y == 0).sum())
    m["parameters"] = count_parameters(model)
    with open(f"{run_dir}/test_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
