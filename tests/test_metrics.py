"""Tests for evaluation metrics."""
import pytest
from src.evaluation.metrics import compute_metrics, expected_calibration_error


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["macro_f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["macro_f1"] == 0.0

    def test_with_probabilities(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert "pr_auc" in metrics
        assert "ece" in metrics
        assert metrics["pr_auc"] > 0.0


class TestECE:
    def test_perfect_calibration(self):
        # If every prediction matches the true frequency, ECE = 0
        y_true = [1] * 50 + [0] * 50
        y_prob = [1.0] * 50 + [0.0] * 50
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.05  # approximately 0

    def test_terrible_calibration(self):
        # Always predict 0.9 but half are actually 0
        y_true = [1] * 50 + [0] * 50
        y_prob = [0.9] * 100
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.3
