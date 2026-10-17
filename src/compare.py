"""Step 5 - Compare architectures on the same test images.

* Table of metrics (with 95% CIs) for every model
* Pairwise DeLong tests: proposed model vs. each baseline
* If several seeds were run, mean +/- SD across seeds

Usage:
    python -m src.compare --seed 42 --proposed densenet121
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_dir, load_config
from .stats import delong_test


def fmt_ci(v, ci):
    return f"{v * 100:.1f} ({ci[0] * 100:.1f}-{ci[1] * 100:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proposed", default="densenet121")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg["output_dir"])
    summary = ensure_dir(out_root / "summary")

    thr = {}
    tp_file = summary / "throughput.json"
    if tp_file.exists():
        tp_rows = json.load(open(tp_file))
        bmax = max(x["batch_size"] for x in tp_rows)
        thr = {r["model"]: r["images_per_sec"] for r in tp_rows if r["batch_size"] == bmax}

    rows, preds = [], {}
    for name in cfg["models"]:
        d = out_root / name / f"seed{args.seed}"
        if not (d / "test_metrics.json").exists():
            print(f"[skip] {name}: no test_metrics.json in {d}")
            continue
        m = json.load(open(d / "test_metrics.json"))
        preds[name] = pd.read_csv(d / "test_predictions.csv")
        ci = m["ci95_bootstrap"]
        rows.append({
            "Model": name,
            "Parameters (M)": round(m["parameters"] / 1e6, 2),
            "Accuracy % (95% CI)": fmt_ci(m["accuracy"], ci["accuracy"]),
            "Precision %": fmt_ci(m["precision"], ci["precision"]),
            "Recall %": fmt_ci(m["sensitivity"], ci["sensitivity"]),
            "Specificity %": fmt_ci(m["specificity"], ci["specificity"]),
            "F1 %": fmt_ci(m["f1"], ci["f1"]),
            "AUC (95% CI, DeLong)": f"{m['auc']:.3f} ({m['auc_ci95_delong'][0]:.3f}-{m['auc_ci95_delong'][1]:.3f})",
            "Throughput (img/s)": thr.get(name, ""),
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        })
    if not rows:
        raise SystemExit("No evaluated models found.")
    table = pd.DataFrame(rows)
    table.to_csv(summary / f"comparison_seed{args.seed}.csv", index=False)
    print(table.to_string(index=False))

    tests = []
    if args.proposed in preds:
        ref = preds[args.proposed].set_index("image")
        for name, p in preds.items():
            if name == args.proposed:
                continue
            p = p.set_index("image").loc[ref.index]
            assert (p.label.values == ref.label.values).all()
            t = delong_test(ref.label.values, ref.prob.values, p.prob.values)
            t.update(comparison=f"{args.proposed} vs {name}")
            tests.append(t)
        tdf = pd.DataFrame(tests)[["comparison", "auc_a", "auc_b", "auc_diff", "z", "p_value"]]
        tdf.to_csv(summary / f"delong_seed{args.seed}.csv", index=False)
        print("\nDeLong tests:\n" + tdf.to_string(index=False))

    # multi-seed aggregation (if available)
    agg = []
    for name in cfg["models"]:
        ms = [json.load(open(f)) for f in sorted((out_root / name).glob("seed*/test_metrics.json"))]
        if len(ms) > 1:
            r = {"Model": name, "n_seeds": len(ms)}
            for k in ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]:
                v = np.array([x[k] for x in ms])
                r[k] = f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}"
            agg.append(r)
    if agg:
        pd.DataFrame(agg).to_csv(summary / "multi_seed_mean_sd.csv", index=False)
        print("\nAcross seeds:\n" + pd.DataFrame(agg).to_string(index=False))


if __name__ == "__main__":
    main()
