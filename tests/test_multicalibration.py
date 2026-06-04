"""Tests for src/calibration/multicalibration.py."""
import numpy as np
import pytest

from src.calibration.multicalibration import (
    MulticalibrationPatcher,
    PerGroupTemperatureScaling,
    apply_global_temperature,
    fit_global_temperature,
)
from src.evaluation.metrics import expected_calibration_error


# ---- Synthetic data helpers ------------------------------------------------

def _make_logits_2class(z: np.ndarray) -> np.ndarray:
    """Build (N, 2) logits with logit_1 - logit_0 = z and logit_0 = 0."""
    return np.stack([np.zeros_like(z), z], axis=1)


def _miscalibrated_overconfident(n: int, scale: float = 3.0, seed: int = 0):
    """Generate (logits, labels) where the model is over-confident.

    True labels follow Bernoulli(σ(z_true)); model logits are inflated by
    ``scale``. Larger ``scale`` → more over-confidence → worse ECE before
    calibration.
    """
    rng = np.random.default_rng(seed)
    z_true = rng.standard_normal(n) * 0.8
    p_true = 1.0 / (1.0 + np.exp(-z_true))
    labels = (rng.random(n) < p_true).astype(np.int64)
    logits = _make_logits_2class(z_true * scale)
    return logits, labels


# ---- Global TS -------------------------------------------------------------

class TestGlobalTemperatureScaling:
    def test_overconfident_T_above_one(self):
        logits, labels = _miscalibrated_overconfident(2000, scale=3.0)
        T = fit_global_temperature(logits, labels)
        assert T > 1.0  # over-confident logits → T > 1 to soften them

    def test_underconfident_T_below_one(self):
        # Apply scale=0.3 → model is under-confident
        logits, labels = _miscalibrated_overconfident(2000, scale=0.3)
        T = fit_global_temperature(logits, labels)
        assert T < 1.0

    def test_calibrated_T_near_one(self):
        logits, labels = _miscalibrated_overconfident(5000, scale=1.0)
        T = fit_global_temperature(logits, labels)
        assert 0.7 < T < 1.5

    def test_apply_returns_probabilities(self):
        logits = np.array([[0.0, 2.0], [0.0, -1.0]])
        probs = apply_global_temperature(logits, T=1.0)
        assert probs.shape == (2,)
        assert np.all((0 <= probs) & (probs <= 1))

    def test_empty_input_returns_default(self):
        T = fit_global_temperature(np.zeros((0, 2)), np.zeros(0))
        assert T == 1.0


# ---- Per-group TS ----------------------------------------------------------

class TestPerGroupTemperatureScaling:
    def test_fit_then_transform_shapes(self):
        logits, labels = _miscalibrated_overconfident(500, scale=2.0)
        groups = np.zeros(len(logits), dtype=np.int64)  # all in group 0
        cal = PerGroupTemperatureScaling(num_groups=2)
        cal.fit(logits, groups, labels)
        probs = cal.transform_probs(logits, groups)
        assert probs.shape == (len(logits),)
        assert np.all((0 <= probs) & (probs <= 1))

    def test_per_group_different_T(self):
        """Two groups with different over-confidence → different fitted T_g."""
        rng = np.random.default_rng(42)
        n = 1000
        # Group 0: heavily over-confident (scale=4)
        z_true_0 = rng.standard_normal(n) * 0.8
        labels_0 = (rng.random(n) < 1 / (1 + np.exp(-z_true_0))).astype(int)
        logits_0 = _make_logits_2class(z_true_0 * 4.0)
        # Group 1: lightly over-confident (scale=1.2)
        z_true_1 = rng.standard_normal(n) * 0.8
        labels_1 = (rng.random(n) < 1 / (1 + np.exp(-z_true_1))).astype(int)
        logits_1 = _make_logits_2class(z_true_1 * 1.2)

        logits = np.vstack([logits_0, logits_1])
        labels = np.concatenate([labels_0, labels_1])
        groups = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)

        cal = PerGroupTemperatureScaling(num_groups=2, min_group_count=50).fit(
            logits, groups, labels
        )
        state = cal.export_state()
        # Group 0 should require a larger T to soften its logits
        assert state["T_per_group"][0] > state["T_per_group"][1]
        assert 0 in state["fitted_groups"]
        assert 1 in state["fitted_groups"]

    def test_sparse_group_falls_back_to_global(self):
        """A group with fewer than min_group_count examples gets global T."""
        rng = np.random.default_rng(0)
        # Group 0 well-populated
        z0 = rng.standard_normal(500) * 0.8
        l0 = (rng.random(500) < 1 / (1 + np.exp(-z0))).astype(int)
        log0 = _make_logits_2class(z0 * 2.0)
        # Group 1: only 5 examples
        z1 = rng.standard_normal(5) * 0.8
        l1 = (rng.random(5) < 1 / (1 + np.exp(-z1))).astype(int)
        log1 = _make_logits_2class(z1 * 2.0)

        logits = np.vstack([log0, log1])
        labels = np.concatenate([l0, l1])
        groups = np.concatenate([np.zeros(500), np.ones(5)]).astype(np.int64)

        cal = PerGroupTemperatureScaling(num_groups=2, min_group_count=30).fit(
            logits, groups, labels
        )
        state = cal.export_state()
        assert 1 not in state["fitted_groups"]
        # Group 1's T should equal the default (global) T
        assert state["T_per_group"][1] == state["default_T"]

    def test_improves_ece_on_overconfident_data(self):
        """End-to-end: fit on dev, transform on test, ECE should decrease."""
        # Two groups with different miscalibration regimes
        rng = np.random.default_rng(7)
        n_train, n_test = 800, 800
        # Same data-generating process across train/test
        def gen(n):
            z = rng.standard_normal(n) * 0.8
            l = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
            log_overconf = _make_logits_2class(z * 3.0)
            return log_overconf, l

        log_dev, lab_dev = gen(n_train)
        log_test, lab_test = gen(n_test)
        groups_dev = np.zeros(n_train, dtype=np.int64)
        groups_test = np.zeros(n_test, dtype=np.int64)

        # Pre-calibration ECE on test
        from scipy.special import expit
        z_test = log_test[:, 1] - log_test[:, 0]
        pre_probs = expit(z_test)
        pre_ece = expected_calibration_error(lab_test.tolist(), pre_probs.tolist())

        cal = PerGroupTemperatureScaling(num_groups=1).fit(log_dev, groups_dev, lab_dev)
        post_probs = cal.transform_probs(log_test, groups_test)
        post_ece = expected_calibration_error(lab_test.tolist(), post_probs.tolist())

        assert post_ece < pre_ece, f"ECE should decrease: pre={pre_ece:.4f}, post={post_ece:.4f}"

    def test_invalid_group_id_raises(self):
        cal = PerGroupTemperatureScaling(num_groups=2)
        with pytest.raises(ValueError):
            cal.fit(
                _make_logits_2class(np.zeros(10)),
                np.full(10, 5, dtype=np.int64),  # group 5 out of [0, 2)
                np.zeros(10),
            )

    def test_transform_before_fit_raises(self):
        cal = PerGroupTemperatureScaling(num_groups=2)
        with pytest.raises(RuntimeError):
            cal.transform_probs(_make_logits_2class(np.zeros(3)), np.zeros(3, dtype=np.int64))


# ---- Multicalibration patcher ---------------------------------------------

class TestMulticalibrationPatcher:
    def test_fit_then_transform_shapes(self):
        logits, labels = _miscalibrated_overconfident(500, scale=2.0)
        groups = np.zeros(len(logits), dtype=np.int64)
        cal = MulticalibrationPatcher(num_groups=2, n_bins=5)
        cal.fit(logits, groups, labels)
        probs = cal.transform_probs(logits, groups)
        assert probs.shape == (len(logits),)
        assert np.all((0 <= probs) & (probs <= 1))

    def test_improves_ece_on_overconfident_data(self):
        logits, labels = _miscalibrated_overconfident(2000, scale=3.0)
        groups = np.zeros(len(logits), dtype=np.int64)

        from scipy.special import expit
        z = logits[:, 1] - logits[:, 0]
        pre_ece = expected_calibration_error(labels.tolist(), expit(z).tolist())

        cal = MulticalibrationPatcher(num_groups=1, n_bins=10, min_bin_count=20).fit(
            logits, groups, labels
        )
        post = cal.transform_probs(logits, groups)
        post_ece = expected_calibration_error(labels.tolist(), post.tolist())
        assert post_ece < pre_ece

    def test_already_calibrated_unchanged(self):
        """If the model is already well-calibrated, the patcher should not change much."""
        rng = np.random.default_rng(11)
        n = 1000
        z = rng.standard_normal(n)
        labels = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        logits = _make_logits_2class(z)
        groups = np.zeros(n, dtype=np.int64)
        cal = MulticalibrationPatcher(num_groups=1, n_bins=10, tol=0.05).fit(
            logits, groups, labels
        )
        state = cal.export_state()
        # Most cells should be unpatched (residual under tol)
        n_patched = sum(len(v) for v in state["corrections"].values())
        assert n_patched <= 5  # very few patches needed

    def test_transform_before_fit_raises(self):
        cal = MulticalibrationPatcher(num_groups=2, n_bins=5)
        with pytest.raises(RuntimeError):
            cal.transform_probs(_make_logits_2class(np.zeros(3)), np.zeros(3, dtype=np.int64))

    def test_export_state_shape(self):
        cal = MulticalibrationPatcher(num_groups=2, n_bins=5)
        # Before fit
        st = cal.export_state()
        assert st["fitted"] is False

        logits, labels = _miscalibrated_overconfident(500, scale=2.0)
        groups = np.zeros(len(logits), dtype=np.int64)
        cal.fit(logits, groups, labels)
        st = cal.export_state()
        assert st["fitted"] is True
        assert st["n_iterations_used"] >= 1
        assert "corrections" in st
