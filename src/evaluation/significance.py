"""Statistical-significance tests for paired model comparisons.

Implements the paired bootstrap test recommended by
Dror, Baumer, Shlomov & Reichart, ACL 2018,
"The Hitchhiker's Guide to Testing Statistical Significance in NLP".

The function `paired_bootstrap` resamples (with replacement) the per-example
predictions of two models on the same test set, recomputes a metric on each
bootstrap sample, and reports a confidence interval on the difference plus
a two-sided p-value for H0: metric(A) == metric(B).

Multi-seed extension: when both models have multiple seeds, we follow the
standard practice of pooling predictions across seeds (one model = the
ensemble's per-example mean prediction OR a randomly-selected seed; default:
mean of probabilities). Both behaviours are supported.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 (positive + negative averaged), implemented inline so that the
    bootstrap loop does not pay sklearn's per-call overhead 10,000 times."""
    f1s = []
    for c in (0, 1):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = (2 * tp + fp + fn)
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _f1_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = (2 * tp + fp + fn)
    return 2 * tp / denom if denom > 0 else 0.0


METRIC_FNS: dict[str, Callable] = {
    # classification metrics — take y_true, y_pred
    "macro_f1":     lambda yt, yp, pp: _f1_macro(yt, yp),
    "f1_positive":  lambda yt, yp, pp: _f1_positive(yt, yp),
    # calibration metrics — take y_true, y_prob (y_pred unused)
    "ece":          lambda yt, yp, pp: expected_calibration_error(yt.tolist(), pp.tolist(), n_bins=15),
    "brier_score":  lambda yt, yp, pp: brier_score(yt.tolist(), pp.tolist()),
    "nll":          lambda yt, yp, pp: negative_log_likelihood(yt.tolist(), pp.tolist()),
}


def paired_bootstrap(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    y_prob_a: Sequence[float] | None = None,
    y_prob_b: Sequence[float] | None = None,
    metric: str = "macro_f1",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 12345,
) -> dict:
    """Two-sided paired bootstrap on the difference metric(A) - metric(B).

    The two models must have been evaluated on the SAME test set in the SAME
    order; the function does not check this for you.

    Args:
        y_true: ground-truth labels, length N.
        y_pred_a, y_pred_b: hard-label predictions of model A and B.
        y_prob_a, y_prob_b: predicted positive-class probabilities (only
            required for calibration metrics like ece / brier_score / nll).
        metric: key into METRIC_FNS.
        n_resamples: number of bootstrap iterations (Dror+18 recommend >=10k).
        confidence_level: e.g. 0.95 for a 95% CI on the difference.
        rng_seed: for reproducibility.

    Returns:
        Dict with the observed difference, its CI bounds, the bootstrap
        p-value (proportion of resamples in which the sign of the difference
        flipped relative to the observed direction, doubled for two-sidedness),
        plus the per-model metric values for context.
    """
    if metric not in METRIC_FNS:
        raise ValueError(f"unknown metric '{metric}'; choose from {list(METRIC_FNS)}")

    yt = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)
    pa = np.asarray(y_prob_a) if y_prob_a is not None else np.zeros_like(yt, dtype=float)
    pb = np.asarray(y_prob_b) if y_prob_b is not None else np.zeros_like(yt, dtype=float)

    if not (len(yt) == len(ya) == len(yb) == len(pa) == len(pb)):
        raise ValueError("y_true, y_pred_*, y_prob_* must have identical length")

    fn = METRIC_FNS[metric]
    obs_a = fn(yt, ya, pa)
    obs_b = fn(yt, yb, pb)
    obs_delta = obs_a - obs_b

    rng = np.random.default_rng(rng_seed)
    n = len(yt)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya_k, yb_k = ya[idx], yb[idx]
        pa_k, pb_k = pa[idx], pb[idx]
        yt_k = yt[idx]
        deltas[k] = fn(yt_k, ya_k, pa_k) - fn(yt_k, yb_k, pb_k)

    alpha = 1 - confidence_level
    ci_lo = float(np.percentile(deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Two-sided bootstrap p-value: probability under the bootstrap distribution
    # that the difference's sign disagrees with the observed direction. Doubled
    # for two-sided. Bounded in [1/n_resamples, 1] to avoid p=0 artefacts.
    if obs_delta >= 0:
        p_one_sided = float(np.mean(deltas <= 0))
    else:
        p_one_sided = float(np.mean(deltas >= 0))
    p_value = min(1.0, max(1.0 / n_resamples, 2 * p_one_sided))

    return {
        "metric": metric,
        "metric_a": obs_a,
        "metric_b": obs_b,
        "delta": obs_delta,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "confidence_level": confidence_level,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "n_examples": int(n),
        "rng_seed": rng_seed,
    }


def load_predictions_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load `test_predictions.csv` (written by scripts/train.py) into arrays.

    Expected columns: text, true_label, predicted_label, confidence, keyword.
    """
    import pandas as pd

    df = pd.read_csv(path)
    return {
        "y_true": df["true_label"].to_numpy(),
        "y_pred": df["predicted_label"].to_numpy(),
        "y_prob": df["confidence"].to_numpy(),
        "keyword": df["keyword"].to_numpy() if "keyword" in df.columns else None,
    }


def average_seed_predictions(
    seed_paths: list[Path],
    decision_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Pool predictions across seeds by averaging the per-example positive-class
    probabilities, then thresholding at `decision_threshold` to recover hard
    labels. The y_true vector must match across all seeds (same test split),
    enforced by an exact-equality check.
    """
    if not seed_paths:
        raise ValueError("seed_paths is empty")

    yt0 = None
    probs_list = []
    for p in seed_paths:
        d = load_predictions_csv(p)
        if yt0 is None:
            yt0 = d["y_true"]
        elif not np.array_equal(yt0, d["y_true"]):
            raise ValueError(
                f"y_true mismatch between seeds at {p}; check that the splits match"
            )
        probs_list.append(d["y_prob"])

    avg_prob = np.mean(np.stack(probs_list, axis=0), axis=0)
    return {
        "y_true": yt0,
        "y_pred": (avg_prob >= decision_threshold).astype(int),
        "y_prob": avg_prob,
    }
