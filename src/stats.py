"""Statistics: classification metrics, DeLong's test (fast O(n log n) version,
Sun & Xu 2014) and bootstrap confidence intervals.

This module depends only on numpy / scipy / scikit-learn.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp + tn) / len(y_true), sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, auc=float(roc_auc_score(y_true, y_prob)), threshold=threshold)


# ---------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: (k, n) with the m positive examples first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _prep(y_true, *probs):
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives first
    m = int(y.sum())
    p = np.vstack([np.asarray(q, dtype=float)[order] for q in probs])
    return p, m


def delong_auc_ci(y_true, y_prob, alpha: float = 0.95):
    p, m = _prep(y_true, y_prob)
    aucs, cov = _fast_delong(p, m)
    auc, se = float(aucs[0]), float(np.sqrt(cov[0, 0]))
    z = sps.norm.ppf(1 - (1 - alpha) / 2)
    return auc, (max(0.0, auc - z * se), min(1.0, auc + z * se)), se


def delong_test(y_true, prob_a, prob_b) -> dict:
    """Two-sided DeLong test for two correlated ROC curves (same test cases)."""
    p, m = _prep(y_true, prob_a, prob_b)
    aucs, cov = _fast_delong(p, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var)
    pval = 2 * sps.norm.sf(abs(z))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), auc_diff=float(diff), z=float(z), p_value=float(pval))


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(y_true, y_prob, threshold=0.5, iters=2000, seed=0, alpha=0.95) -> dict:
    """Percentile bootstrap CIs for every metric in binary_metrics()."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    samples = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        m = binary_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = (1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}
