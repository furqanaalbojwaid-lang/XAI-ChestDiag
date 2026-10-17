"""Step 1 - Build binary labels and a strict patient-level 70/10/20 split.

Label rule (emergency triage):
    "No Finding"            -> 0 (Normal)
    any of the 14 findings  -> 1 (Infected / Abnormal)

All images of a patient go to exactly one partition, so no patient appears
in more than one of train / val / test (no data leakage).

Usage:
    python -m src.prepare_splits --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .common import ensure_dir, load_config


def make_splits(labels_csv: str, train: float, val: float, test: float, seed: int) -> pd.DataFrame:
    assert abs(train + val + test - 1.0) < 1e-6, "split fractions must sum to 1"
    df = pd.read_csv(labels_csv)
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "findings", "Patient ID": "patient_id"})
    df = df[["image", "findings", "patient_id", "View Position"]].copy()
    df["label"] = (df["findings"].str.strip() != "No Finding").astype(int)

    patients = df["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    part = {}
    for i, pid in enumerate(patients):
        part[pid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    df["split"] = df["patient_id"].map(part)
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        out[s] = {
            "images": int(len(d)),
            "patients": int(d.patient_id.nunique()),
            "abnormal": int(d.label.sum()),
            "normal": int((d.label == 0).sum()),
            "image_fraction": round(len(d) / len(df), 4),
        }
    out["total"] = {
        "images": int(len(df)),
        "patients": int(df.patient_id.nunique()),
        "abnormal": int(df.label.sum()),
        "normal": int((df.label == 0).sum()),
    }
    # leakage check
    sets = {s: set(df[df.split == s].patient_id) for s in ["train", "val", "test"]}
    overlap = len(sets["train"] & sets["val"]) + len(sets["train"] & sets["test"]) + len(sets["val"] & sets["test"])
    out["patient_overlap_between_splits"] = overlap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg["split"]
    df = make_splits(cfg["data"]["labels_csv"], s["train"], s["val"], s["test"], s["seed"])
    out_dir = ensure_dir(cfg["data"]["split_dir"])
    df.to_csv(out_dir / "all_splits.csv", index=False)
    for name in ["train", "val", "test"]:
        df[df.split == name].to_csv(out_dir / f"{name}.csv", index=False)
    stats = summarize(df)
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    assert stats["patient_overlap_between_splits"] == 0


if __name__ == "__main__":
    main()
