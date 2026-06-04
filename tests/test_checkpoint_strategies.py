"""Tests for the checkpoint-selection strategies (calibration bake-off Layer 1).

We avoid a real DataLoader for speed; instead we exercise:
  - score(...) returns the right value for each strategy
  - on_epoch_end / finalize don't crash and behave correctly
  - SWA averages the right snapshots
  - TemperatureScaling fits a positive scalar and changes transform_logits
  - build_strategy parses YAML-style specs correctly
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.training.checkpoint_strategies import (
    BestF1Strategy,
    BestF1PositiveStrategy,
    CompositeF1ECEStrategy,
    SWAStrategy,
    TemperatureScalingStrategy,
    build_strategy,
)


# -------- score-only --------------------------------------------------------


def test_best_f1_score():
    s = BestF1Strategy()
    assert s.score({"macro_f1": 0.8, "f1_positive": 0.6, "ece": 0.1}) == 0.8


def test_best_f1_positive_score():
    s = BestF1PositiveStrategy()
    assert s.score({"macro_f1": 0.8, "f1_positive": 0.6, "ece": 0.1}) == 0.6


def test_composite_score_alpha_zero_is_best_f1():
    s = CompositeF1ECEStrategy(alpha=0.0)
    metrics = {"macro_f1": 0.8, "f1_positive": 0.6, "ece": 0.1}
    assert s.score(metrics) == 0.8


def test_composite_score_alpha_penalises_ece():
    s = CompositeF1ECEStrategy(alpha=2.0)
    # 0.8 - 2.0 * 0.1 = 0.6
    assert math.isclose(s.score({"macro_f1": 0.8, "ece": 0.1}), 0.6)


def test_composite_rejects_negative_alpha():
    import pytest
    with pytest.raises(ValueError):
        CompositeF1ECEStrategy(alpha=-1.0)


def test_composite_name_includes_alpha():
    s = CompositeF1ECEStrategy(alpha=0.5)
    assert "0.5" in s.name


# -------- SWA ---------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """Linear layer just so we can inspect state_dict averaging."""
    def __init__(self, init_value: float = 0.0):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(init_value)


def test_swa_rejects_k_below_2():
    import pytest
    with pytest.raises(ValueError):
        SWAStrategy(k=1)


def test_swa_keeps_only_last_k_snapshots():
    s = SWAStrategy(k=2)
    for epoch_idx, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        m = _TinyModel(init_value=val)
        s.on_epoch_end(epoch_idx, m, {"macro_f1": 0.5}, dev_loader=None, device="cpu")
    # After 5 epochs we should have only the last K=2 snapshots
    assert len(s._snapshots) == 2
    # Snapshots correspond to epochs 4 and 5 (1-indexed)
    assert s._snapshot_epochs == [4, 5]


def test_swa_finalize_averages_weights(tmp_path: Path):
    s = SWAStrategy(k=3)
    final_model = _TinyModel(init_value=0.0)
    # Push three snapshots with weights 1, 3, 5  (mean = 3)
    for epoch_idx, val in enumerate([1.0, 3.0, 5.0]):
        m = _TinyModel(init_value=val)
        s.on_epoch_end(epoch_idx, m, {"macro_f1": 0.5}, dev_loader=None, device="cpu")
    out = s.finalize(final_model, save_dir=tmp_path, dev_loader=None, device="cpu")
    assert torch.allclose(out.linear.weight, torch.full((2, 2), 3.0))
    # Sidecar metadata file written
    assert (tmp_path / "swa_meta.json").exists()
    assert (tmp_path / "swa_averaged.pt").exists()


def test_swa_with_one_snapshot_does_nothing(tmp_path: Path):
    """SWA with k=2 but only 1 snapshot collected -> falls back to current weights."""
    s = SWAStrategy(k=2)
    m = _TinyModel(init_value=7.0)
    s.on_epoch_end(0, m, {"macro_f1": 0.5}, dev_loader=None, device="cpu")
    # Don't call on_epoch_end again. finalize should be a no-op.
    out = s.finalize(m, save_dir=tmp_path, dev_loader=None, device="cpu")
    # Weights unchanged
    assert torch.allclose(out.linear.weight, torch.full((2, 2), 7.0))


# -------- Temperature scaling ---------------------------------------------


def test_temperature_scaling_default_unfitted_is_identity():
    s = TemperatureScalingStrategy()
    logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    out = s.transform_logits(logits)
    assert torch.equal(out, logits)


def test_temperature_scaling_after_fit_divides_logits():
    s = TemperatureScalingStrategy()
    s.fitted_T = 2.0
    s._is_fitted = True
    logits = torch.tensor([[2.0, 4.0]])
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.equal(s.transform_logits(logits), expected)


# -------- Factory ---------------------------------------------------------


def test_build_strategy_default_is_best_f1():
    s = build_strategy(None)
    assert isinstance(s, BestF1Strategy)


def test_build_strategy_from_string():
    s = build_strategy("best_f1_positive")
    assert isinstance(s, BestF1PositiveStrategy)


def test_build_strategy_composite_with_alpha():
    s = build_strategy({"name": "composite_f1_ece", "alpha": 0.5})
    assert isinstance(s, CompositeF1ECEStrategy)
    assert s.alpha == 0.5


def test_build_strategy_swa_with_k():
    s = build_strategy({"name": "swa", "k": 3})
    assert isinstance(s, SWAStrategy)
    assert s.k == 3


def test_build_strategy_temperature_scaling():
    s = build_strategy({"name": "temperature_scaling"})
    assert isinstance(s, TemperatureScalingStrategy)


def test_build_strategy_unknown_name_raises():
    import pytest
    with pytest.raises(ValueError):
        build_strategy({"name": "nonexistent_strategy"})
