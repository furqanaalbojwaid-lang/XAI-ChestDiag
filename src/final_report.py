"""Final tables + figures for the paper (CPU only, reads saved predictions).

* Ablation table of the proposed model variants (validation-selected, test-reported)
* Comparison of the final proposed model vs. the four baseline CNNs
* DeLong tests: final proposed model vs. every other model / variant
* ROC curves, confusion matrix and training curves of the final model

Usage:
    python -m src.final_report --config configs/kaggle.yaml \
        --baselines densenet121 vgg16 inception_v3 efficientnet_b4 \
        --ablation densenet121 a1_multitask a2_mt_cbam a3_mt_cbam_modern a4_full_384 \
        --final a4_full_384
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
from .plots import confusion_matrix_fig, save
from .stats import delong_test

LABEL = {"densenet121": "DenseNet121 (baseline)", "vgg16": "VGG16", "inception_v3": "InceptionV3",
         "efficientnet_b4": "EfficientNet-B4", "a1_multitask": "+ Multi-task head",
         "a2_mt_cbam": "+ Multi-task + CBAM", "a3_mt_cbam_modern": "+ Modern training recipe",
         "a4_full_384": "+ 384 px input (full XAI-ChestDiag)",
         "a5_xrv_384": "+ Chest-X-ray pre-training (CheXpert init)"}

# Ensembles are fixed IN ADVANCE (no member selection on the test set): equal-weight mean
# of the member probabilities (TTA predictions used where available).
ENSEMBLES = {"XAI-ChestDiag-Ens (proposed variants)": ["a3_mt_cbam_modern", "a4_full_384", "a5_xrv_384"],
             "Ens (all models)": ["a3_mt_cbam_modern", "a4_full_384", "a5_xrv_384",
                                  "densenet121", "vgg16", "inception_v3", "efficientnet_b4"]}


def ensemble_eval(root, members, seed, ec_thr=0.5, n_boot=2000):
    from .stats import binary_metrics, bootstrap_ci, delong_auc_ci
    rs = [(m, load_run(root, m, seed, tta=True)) for m in members]
    rs = [(m, r) for m, r in rs if r is not None]
    if len(rs) < 2:
        return None
    idx = rs[0][1]["pred"].index
    y = rs[0][1]["pred"].label.values
    prob = np.mean([r["pred"].loc[idx].prob.values for _, r in rs], axis=0)
    m = binary_metrics(y, prob, ec_thr)
    auc, (lo, hi), _ = delong_auc_ci(y, prob)
    m["auc_ci95_delong"] = [lo, hi]
    m["ci95_bootstrap"] = bootstrap_ci(y, prob, ec_thr, n_boot, seed=seed)
    # Youden threshold from VALIDATION predictions, only if every member saved them
    vfiles = [r["dir"] / ("val_predictions_tta.csv" if r["tta"] else "val_predictions.csv") for _, r in rs]
    if all(f.exists() for f in vfiles):
        from sklearn.metrics import roc_curve as _rc
        vs = [pd.read_csv(f).set_index("image") for f in vfiles]
        vy = vs[0].label.values
        vp = np.mean([v.loc[vs[0].index].prob.values for v in vs], axis=0)
        fpr, tpr, thr = _rc(vy, vp)
        t = float(thr[np.argmax(tpr - fpr)])
        m["youden"] = {**binary_metrics(y, prob, t), "ci95_bootstrap": bootstrap_ci(y, prob, t, n_boot, seed=seed),
                       "threshold_chosen_on": "validation"}
    m["parameters"] = sum(r["metrics"]["parameters"] for _, r in rs)
    m["image_size"] = "/".join(str(r["metrics"].get("image_size", 224)) for _, r in rs)
    pred = rs[0][1]["pred"].loc[idx][["label"]].copy()
    pred["prob"] = prob
    return dict(metrics=m, pred=pred, dir=None, tta=any(r["tta"] for _, r in rs), members=[mm for mm, _ in rs])


def load_run(root: Path, run: str, seed: int, tta: bool):
    d = root / run / f"seed{seed}"
    sfx = "_tta" if tta and (d / "test_metrics_tta.json").exists() else ""
    if not (d / f"test_metrics{sfx}.json").exists():
        return None
    return dict(metrics=json.load(open(d / f"test_metrics{sfx}.json")),
                pred=pd.read_csv(d / f"test_predictions{sfx}.csv").set_index("image"), dir=d, tta=bool(sfx))


def row(name, r, use_youden=False):
    m = r["metrics"]["youden"] if use_youden and "youden" in r["metrics"] else r["metrics"]
    ci = m.get("ci95_bootstrap", r["metrics"]["ci95_bootstrap"])
    f = lambda k: f"{m[k] * 100:.1f} ({ci[k][0] * 100:.1f}-{ci[k][1] * 100:.1f})"
    lo, hi = r["metrics"]["auc_ci95_delong"]
    return {"Model": LABEL.get(name, name), "Params (M)": round(r["metrics"]["parameters"] / 1e6, 2),
            "Input": r["metrics"].get("image_size", 224), "TTA": "yes" if r["tta"] else "no",
            "Threshold": f"{m['threshold']:.3f}" + (" (Youden, val)" if use_youden else ""),
            "Accuracy %": f("accuracy"), "Sensitivity %": f("sensitivity"), "Specificity %": f("specificity"),
            "Precision %": f("precision"), "F1 %": f("f1"),
            "AUC (95% CI)": f"{r['metrics']['auc']:.3f} ({lo:.3f}-{hi:.3f})",
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines", nargs="+", default=["densenet121", "vgg16", "inception_v3", "efficientnet_b4"])
    ap.add_argument("--ablation", nargs="+", default=["densenet121", "a1_multitask", "a2_mt_cbam",
                                                       "a3_mt_cbam_modern", "a4_full_384", "a5_xrv_384"])
    ap.add_argument("--final", default="a4_full_384")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    out = ensure_dir(root / "summary")
    figs = ensure_dir(root / "figures")

    # ---------------- ablation (no TTA, threshold 0.5 -> isolates each training change)
    abl = []
    for run in args.ablation:
        r = load_run(root, run, args.seed, tta=False)
        if r is None:
            print("[skip]", run); continue
        abl.append(row(run, r))
    final_plain = load_run(root, args.final, args.seed, tta=False)
    final_tta = load_run(root, args.final, args.seed, tta=True)
    if final_tta is not None and final_tta["tta"]:
        abl.append({**row(args.final, final_tta), "Model": "+ Test-time augmentation"})
        abl.append({**row(args.final, final_tta, use_youden=True), "Model": "+ Youden threshold (validation)"})
    pd.DataFrame(abl).to_csv(out / "ablation_table.csv", index=False)
    print("\nABLATION\n" + pd.DataFrame(abl).to_string(index=False))

    # ---------------- final comparison vs baselines
    final = final_tta or final_plain
    comp, preds = [], {}
    for run in args.baselines:
        r = load_run(root, run, args.seed, tta=False)
        if r is not None:
            comp.append(row(run, r)); preds[LABEL.get(run, run)] = r["pred"]
    if final is not None:
        comp.append({**row(args.final, final), "Model": "XAI-ChestDiag (proposed)"})
        comp.append({**row(args.final, final, use_youden=True), "Model": "XAI-ChestDiag (proposed, Youden thr.)"})
        preds["XAI-ChestDiag (proposed)"] = final["pred"]
    pd.DataFrame(comp).to_csv(out / "final_comparison.csv", index=False)
    print("\nCOMPARISON\n" + pd.DataFrame(comp).to_string(index=False))

    # ---------------- DeLong: final vs everything else
    if final is not None:
        ref = final["pred"]
        tests = []
        others = {LABEL.get(r, r): r for r in dict.fromkeys(args.baselines + args.ablation) if r != args.final}
        for lab, run in others.items():
            r = load_run(root, run, args.seed, tta=False)
            if r is None:
                continue
            p = r["pred"].loc[ref.index]
            assert (p.label.values == ref.label.values).all()
            t = delong_test(ref.label.values, ref.prob.values, p.prob.values)
            tests.append(dict(comparison=f"XAI-ChestDiag vs {lab}", auc_proposed=t["auc_a"], auc_other=t["auc_b"],
                              delta_auc=t["auc_diff"], z=t["z"], p_value=t["p_value"]))
        pd.DataFrame(tests).to_csv(out / "delong_final.csv", index=False)
        print("\nDELONG\n" + pd.DataFrame(tests).to_string(index=False))

        # ---------------- pre-specified ensembles
        ens_rows, ens_tests = [], []
        for name, members in ENSEMBLES.items():
            e = ensemble_eval(root, members, args.seed)
            if e is None:
                print("[skip ensemble]", name); continue
            ens_rows.append({**row(name, e), "Model": f"{name} [{', '.join(e['members'])}]"})
            if "youden" in e["metrics"]:
                ens_rows.append({**row(name, e, use_youden=True), "Model": f"{name}, Youden thr."})
            p = e["pred"].loc[ref.index]
            t = delong_test(ref.label.values, p.prob.values, ref.prob.values)
            ens_tests.append(dict(comparison=f"{name} vs XAI-ChestDiag (single)", auc_ens=t["auc_a"],
                                  auc_single=t["auc_b"], delta_auc=t["auc_diff"], z=t["z"], p_value=t["p_value"]))
            for run in ["densenet121"]:
                r = load_run(root, run, args.seed, tta=False)
                if r is not None:
                    t = delong_test(ref.label.values, p.prob.values, r["pred"].loc[ref.index].prob.values)
                    ens_tests.append(dict(comparison=f"{name} vs {LABEL[run]}", auc_ens=t["auc_a"],
                                          auc_single=t["auc_b"], delta_auc=t["auc_diff"], z=t["z"], p_value=t["p_value"]))
            if name.startswith("XAI-ChestDiag-Ens"):
                preds[name] = e["pred"]
        if ens_rows:
            pd.DataFrame(ens_rows).to_csv(out / "ensemble_table.csv", index=False)
            pd.DataFrame(ens_tests).to_csv(out / "delong_ensemble.csv", index=False)
            print("\nENSEMBLES\n" + pd.DataFrame(ens_rows).to_string(index=False))
            print("\nDELONG (ensembles)\n" + pd.DataFrame(ens_tests).to_string(index=False))

        # ---------------- figures
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        colors = ["#7F8C8D", "#2E86C1", "#28B463", "#8E44AD", "#E67E22", "#C0392B"]
        main = "XAI-ChestDiag (proposed)"
        order = [(k, v) for k, v in preds.items() if k != main] + [(main, preds[main])]
        for (lab, p), c in zip(order, colors[-len(order):] if len(order) < len(colors) else colors):
            fpr, tpr, _ = roc_curve(p.label, p.prob)
            from sklearn.metrics import roc_auc_score
            ax.plot(fpr, tpr, lw=2.2 if "proposed" in lab else 1.3, color=c,
                    label=f"{lab}: AUC {roc_auc_score(p.label, p.prob):.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        save(fig, figs / "fig_roc_final")
        confusion_matrix_fig(final["metrics"]["youden"], figs / "fig_confusion_matrix_final")
        h = final["dir"] / "history.csv"
        if h.exists():
            hist = pd.read_csv(h)
            info = json.load(open(final["dir"] / "train_info.json"))
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            axes[0].plot(hist.epoch, hist.train_loss, label="Training", color="#2E86C1")
            axes[0].plot(hist.epoch, hist.val_loss, label="Validation", color="#C0392B")
            axes[0].set_ylabel("Loss"); axes[0].legend(frameon=False)
            axes[1].plot(hist.epoch, hist.val_auc, color="#C0392B")
            axes[1].set_ylabel("Validation AUC")
            for a in axes:
                a.set_xlabel("Epoch"); a.axvline(info["best_epoch"], ls=":", color="#555")
            fig.tight_layout()
            save(fig, figs / "fig_training_curves_final")


if __name__ == "__main__":
    main()
