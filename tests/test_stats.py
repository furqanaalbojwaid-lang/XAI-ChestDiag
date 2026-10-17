"""Sanity tests for the statistics module (run: python -m pytest tests -q)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.stats import binary_metrics, bootstrap_ci, delong_auc_ci, delong_test


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = np.clip(y * 0.35 + rng.normal(0.4, 0.2, n), 0, 1)
    b = np.clip(y * 0.20 + rng.normal(0.4, 0.2, n), 0, 1)
    return y, a, b


def test_delong_auc_matches_sklearn():
    y, a, _ = _data()
    auc, (lo, hi), se = delong_auc_ci(y, a)
    assert abs(auc - roc_auc_score(y, a)) < 1e-9
    assert lo < auc < hi and se > 0


def test_delong_detects_difference_and_symmetry():
    y, a, b = _data()
    t = delong_test(y, a, b)
    assert t["p_value"] < 1e-3 and t["z"] > 0
    t2 = delong_test(y, b, a)
    assert abs(t["z"] + t2["z"]) < 1e-9


def test_delong_identical_models_not_significant():
    y, a, _ = _data()
    rng = np.random.default_rng(1)
    a2 = np.clip(a + rng.normal(0, 0.01, len(a)), 0, 1)
    assert delong_test(y, a, a2)["p_value"] > 0.01


def test_binary_metrics_hand_example():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.3])
    m = binary_metrics(y, p, 0.5)
    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (2, 1, 2, 1)
    assert abs(m["precision"] - 2 / 3) < 1e-12 and abs(m["accuracy"] - 4 / 6) < 1e-12


def test_bootstrap_ci_contains_point_estimate():
    y, a, _ = _data(800)
    m = binary_metrics(y, a)
    ci = bootstrap_ci(y, a, iters=300)
    for k in ["accuracy", "auc", "f1"]:
        assert ci[k][0] <= m[k] <= ci[k][1]
