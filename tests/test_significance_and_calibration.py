"""Tests for the new calibration metrics and the paired-bootstrap significance test.

The numbers below are computed analytically from small fixed inputs and compared
against the implementation. If the test fails, either the formula or the
implementation has drifted.
"""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_diagram,
)
from src.evaluation.significance import paired_bootstrap


# ---------- calibration metrics --------------------------------------------


def test_brier_score_perfect_predictions():
    y_true = [1, 1, 0, 0]
    y_prob = [1.0, 1.0, 0.0, 0.0]
    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_uniform_05():
    # All probabilities = 0.5: per-example squared error = 0.25 always.
    y_true = [1, 0, 1, 0]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    assert math.isclose(brier_score(y_true, y_prob), 0.25)


def test_nll_perfect_predictions():
    y_true = [1, 1, 0, 0]
    y_prob = [1.0, 1.0, 0.0, 0.0]
    # eps clip prevents -inf, so NLL is tiny but not exactly zero.
    assert negative_log_likelihood(y_true, y_prob) < 1e-9


def test_nll_uniform_05():
    # NLL of all-0.5 = -log(0.5) = log(2)
    y_true = [1, 0, 1, 0]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    assert math.isclose(negative_log_likelihood(y_true, y_prob), math.log(2), abs_tol=1e-9)


def test_ece_perfect_calibration_constant_prob():
    # 50% positives, all predictions = 0.5 -> bin accuracy 0.5, bin conf 0.5, gap 0.
    y_true = [1, 0, 1, 0, 1, 0]
    y_prob = [0.5] * 6
    assert expected_calibration_error(y_true, y_prob, n_bins=15) == 0.0


def test_mce_catches_one_bad_bin():
    # Two bins occupied: one perfect, one with confidence 0.9 but accuracy 0.0
    y_true = [1, 1, 0, 0]
    y_prob = [0.1, 0.1, 0.9, 0.9]
    # Bin 0.0–0.067 -> empty; bin 0.067–0.133 occupied; ... bin 0.867–0.933 occupied
    # All-0.9 bin: confidence 0.9, accuracy 0.0 -> gap = 0.9
    assert math.isclose(maximum_calibration_error(y_true, y_prob, n_bins=15), 0.9, abs_tol=1e-9)


def test_reliability_diagram_shape():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.4, 0.3]
    rd = reliability_diagram(y_true, y_prob, n_bins=10)
    assert rd["n_bins"] == 10
    assert len(rd["bin_lower"]) == 10
    assert len(rd["bin_upper"]) == 10
    assert len(rd["bin_count"]) == 10
    assert sum(rd["bin_count"]) == len(y_true)


# ---------- paired bootstrap ------------------------------------------------


def test_paired_bootstrap_identical_models_have_zero_delta():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_prob = rng.random(n)

    res = paired_bootstrap(
        y_true=y_true,
        y_pred_a=y_pred, y_pred_b=y_pred,
        y_prob_a=y_prob, y_prob_b=y_prob,
        metric="macro_f1",
        n_resamples=500,
        rng_seed=42,
    )
    assert res["delta"] == 0.0
    # When the two models are identical, all bootstrap deltas are 0; the
    # one-sided p (proportion with sign opposite to observed direction) is
    # 1.0 because 0 satisfies "<= 0", and the doubled two-sided p caps at 1.0.
    # In short: no evidence against H0, p-value should be at the maximum.
    assert res["p_value"] == 1.0
    # CI on the difference must straddle zero (it should literally be [0, 0]).
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0


def test_paired_bootstrap_detects_clear_difference():
    # Construct a setting where model A is much better than model B.
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, size=n)
    y_pred_a = y_true.copy()  # perfect
    y_pred_b = 1 - y_true     # always wrong
    y_prob_a = y_true.astype(float)
    y_prob_b = 1.0 - y_true.astype(float)

    res = paired_bootstrap(
        y_true=y_true,
        y_pred_a=y_pred_a, y_pred_b=y_pred_b,
        y_prob_a=y_prob_a, y_prob_b=y_prob_b,
        metric="macro_f1",
        n_resamples=2000,
        rng_seed=42,
    )
    assert res["delta"] > 0.5
    assert res["p_value"] < 0.01
    assert res["ci_low"] > 0  # CI strictly above zero


def test_paired_bootstrap_known_metrics():
    """Smoke test that all registered metrics can be evaluated without crashing."""
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_pred_a = rng.integers(0, 2, size=n)
    y_pred_b = rng.integers(0, 2, size=n)
    y_prob_a = rng.random(n)
    y_prob_b = rng.random(n)
    for metric in ("macro_f1", "f1_positive", "ece", "brier_score", "nll"):
        res = paired_bootstrap(
            y_true=y_true,
            y_pred_a=y_pred_a, y_pred_b=y_pred_b,
            y_prob_a=y_prob_a, y_prob_b=y_prob_b,
            metric=metric,
            n_resamples=200,
            rng_seed=0,
        )
        assert "p_value" in res
        assert 0 <= res["p_value"] <= 1
