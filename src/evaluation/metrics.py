"""Evaluation metrics for antisemitism detection.

Includes both classification metrics (F1, P/R, PR-AUC) and calibration metrics
(ECE, MCE, Brier score, NLL, reliability-diagram data). The richer calibration
panel was added on 2026-04-27 in response to peer-review feedback that ECE
alone is binning-sensitive and can hide minority-class miscalibration.
"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] = None,
    n_bins: int = 15,
) -> dict:
    """
    Compute classification + calibration metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probability of positive class (for PR-AUC + calibration)
        n_bins: Number of bins for ECE / MCE / reliability diagram. We use 15
                bins (Guo et al. ICML 2017 default) to stay comparable with the
                calibration literature.

    Returns:
        Dict with macro_f1, precision, recall, f1_positive, f1_negative,
        confusion_matrix, plus (when y_prob is given) pr_auc, ece, mce,
        brier_score, nll, reliability (per-bin breakdown for plotting).
    """
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_negative": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    if y_prob is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        metrics["mce"] = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
        metrics["brier_score"] = brier_score(y_true, y_prob)
        metrics["nll"] = negative_log_likelihood(y_true, y_prob)
        metrics["reliability"] = reliability_diagram(y_true, y_prob, n_bins=n_bins)

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def expected_calibration_error(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies,
    weighted by bin population. Lower is better. ECE=0 means perfectly
    calibrated. Guo et al., ICML 2017.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # right-closed last bin to include 1.0
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.sum() / len(y_true)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def maximum_calibration_error(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 15,
) -> float:
    """
    Maximum Calibration Error (MCE).

    The worst (largest) per-bin gap between confidence and accuracy. Useful as
    a worst-case calibration bound when ECE averages can hide a single very
    miscalibrated bin. Guo et al., ICML 2017.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    gaps = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        gaps.append(abs(bin_accuracy - bin_confidence))

    return float(max(gaps)) if gaps else 0.0


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """
    Brier score: mean squared error between predicted probability and the
    one-hot ground truth. Strictly proper scoring rule. Lower is better.
    Brier 1950.

    Defined for binary classification: BS = (1/N) * sum_i (p_i - y_i)^2
    where p_i in [0,1] is the predicted probability of the positive class.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def negative_log_likelihood(
    y_true: list[int],
    y_prob: list[float],
    eps: float = 1e-12,
) -> float:
    """
    Mean negative log-likelihood (a.k.a. cross-entropy / log loss).

    NLL = -(1/N) sum_i [ y_i log p_i + (1 - y_i) log (1 - p_i) ]

    Strictly proper scoring rule. Lower is better. Probabilities are clipped to
    [eps, 1 - eps] for numerical stability.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def reliability_diagram(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 15,
) -> dict:
    """
    Per-bin breakdown for plotting a reliability diagram.

    Returns a dict with one list per axis:
        bin_lower, bin_upper, bin_mean_confidence, bin_accuracy, bin_count

    For perfectly calibrated models, bin_mean_confidence == bin_accuracy in
    every populated bin (the diagonal).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    out = {
        "bin_lower": [],
        "bin_upper": [],
        "bin_mean_confidence": [],
        "bin_accuracy": [],
        "bin_count": [],
        "n_bins": n_bins,
    }
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n = int(mask.sum())
        out["bin_lower"].append(float(bin_boundaries[i]))
        out["bin_upper"].append(float(bin_boundaries[i + 1]))
        out["bin_count"].append(n)
        if n == 0:
            out["bin_mean_confidence"].append(None)
            out["bin_accuracy"].append(None)
        else:
            out["bin_mean_confidence"].append(float(y_prob[mask].mean()))
            out["bin_accuracy"].append(float(y_true[mask].mean()))
    return out


def per_keyword_metrics(
    y_true: list[int],
    y_pred: list[int],
    keywords: list[str],
) -> dict:
    """Compute F1 breakdown by keyword group."""
    unique_keywords = sorted(set(keywords))
    results = {}

    for kw in unique_keywords:
        mask = [k == kw for k in keywords]
        yt = [y for y, m in zip(y_true, mask) if m]
        yp = [y for y, m in zip(y_pred, mask) if m]

        if len(yt) == 0:
            continue

        results[kw] = {
            "n": len(yt),
            "n_positive": sum(yt),
            "base_rate": sum(yt) / len(yt),
            "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
            "precision": precision_score(yt, yp, pos_label=1, zero_division=0),
            "recall": recall_score(yt, yp, pos_label=1, zero_division=0),
        }

    return results
