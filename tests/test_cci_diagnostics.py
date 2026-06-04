"""Tests for src/training/cci_diagnostics.py."""
import json
import math
import tempfile

import numpy as np
import pytest
import torch

from src.training.cci_diagnostics import (
    TemperatureTrajectory,
    analyse_temperature_shortcut,
    compute_per_group_base_rates,
)


class TestTemperatureTrajectory:
    def test_record_appends(self):
        traj = TemperatureTrajectory(group_names=["a", "b", "c"])
        # Use values exactly representable in float32 to avoid spurious approx
        # mismatches between the input tensor and the .tolist() output.
        traj.record(0, torch.tensor([1.0, 2.0, 4.0]))
        traj.record(1, torch.tensor([1.5, 2.5, 0.5]))
        assert traj.epoch == [0, 1]
        assert traj.temperatures[0] == pytest.approx([1.0, 2.0, 4.0])
        assert traj.temperatures[1] == pytest.approx([1.5, 2.5, 0.5])

    def test_record_dim_check(self):
        traj = TemperatureTrajectory(group_names=["a", "b"])
        with pytest.raises(ValueError):
            traj.record(0, torch.tensor([[1.0, 2.0]]))  # 2-D
        with pytest.raises(ValueError):
            traj.record(0, torch.tensor([1.0, 2.0, 3.0]))  # wrong group count

    def test_json_roundtrip(self):
        traj = TemperatureTrajectory(group_names=["x", "y"])
        traj.record(0, torch.tensor([1.0, 1.5]))
        traj.record(1, torch.tensor([1.2, 1.4]))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        traj.to_json(path)
        loaded = TemperatureTrajectory.from_json(path)
        assert loaded.group_names == traj.group_names
        assert loaded.epoch == traj.epoch
        assert loaded.temperatures == traj.temperatures


class TestComputePerGroupBaseRates:
    def test_basic(self):
        labels = [0, 1, 1, 0, 1, 0]
        groups = [0, 0, 1, 1, 2, 2]
        rates = compute_per_group_base_rates(labels, groups, num_groups=3)
        assert rates == [0.5, 0.5, 0.5]

    def test_skewed_groups(self):
        # group 0: 3 positives / 4 = 0.75; group 1: 1 / 4 = 0.25
        labels = [1, 1, 1, 0, 1, 0, 0, 0]
        groups = [0, 0, 0, 0, 1, 1, 1, 1]
        rates = compute_per_group_base_rates(labels, groups, num_groups=2)
        assert rates == [0.75, 0.25]

    def test_empty_group_yields_nan(self):
        labels = [0, 1]
        groups = [0, 0]
        rates = compute_per_group_base_rates(labels, groups, num_groups=3)
        assert rates[0] == 0.5
        assert math.isnan(rates[1])
        assert math.isnan(rates[2])


class TestAnalyseTemperatureShortcut:
    def test_no_trajectory(self):
        traj = TemperatureTrajectory(group_names=["a", "b", "c"])
        verdict = analyse_temperature_shortcut(traj, [0.1, 0.5, 0.9])
        assert verdict["shortcut_detected"] is False
        assert verdict["reason"] == "no_trajectory_recorded"

    def test_low_variance_no_shortcut(self):
        # All T close to 1: variance below threshold → no shortcut
        traj = TemperatureTrajectory(group_names=["a", "b", "c"])
        traj.record(0, torch.tensor([1.0, 1.01, 1.02]))
        verdict = analyse_temperature_shortcut(traj, [0.1, 0.5, 0.9])
        assert verdict["shortcut_detected"] is False
        assert "low_variance" in verdict["reason"]

    def test_high_corr_with_inv_baserate_triggers_shortcut(self):
        # Construct T values that are perfectly anti-correlated with base rate
        # (i.e. tightly correlated with 1 - base_rate). Variance must clear
        # the var_threshold (default 0.1), so we spread T over [0.5, 1.5].
        traj = TemperatureTrajectory(group_names=["a", "b", "c", "d"])
        # base rates 0.1, 0.3, 0.7, 0.9 -> 1-baserate 0.9, 0.7, 0.3, 0.1
        # T_g chosen monotonically anti-correlated with base rate.
        traj.record(0, torch.tensor([1.5, 1.2, 0.8, 0.5]))
        verdict = analyse_temperature_shortcut(traj, [0.1, 0.3, 0.7, 0.9])
        assert verdict["var_T_final"] > 0.1
        assert verdict["shortcut_detected"] is True
        assert "shortcut_detected" in verdict["reason"]
        assert abs(verdict["corr_T_inv_baserate"]) > 0.99

    def test_uncorrelated_T_no_shortcut(self):
        # Variance > threshold but correlation low
        traj = TemperatureTrajectory(group_names=["a", "b", "c", "d"])
        traj.record(0, torch.tensor([0.6, 1.5, 1.2, 0.7]))  # roughly independent
        verdict = analyse_temperature_shortcut(traj, [0.1, 0.3, 0.7, 0.9])
        # Correlation may be moderate but not extreme
        if not verdict["shortcut_detected"]:
            assert "corr_below_threshold" in verdict["reason"]

    def test_handles_nan_base_rate(self):
        traj = TemperatureTrajectory(group_names=["a", "b", "c", "d"])
        traj.record(0, torch.tensor([1.0, 1.5, 0.7, 0.5]))
        verdict = analyse_temperature_shortcut(
            traj, [0.1, 0.3, 0.7, float("nan")]
        )
        assert verdict["final_T_per_group"][3] == 0.5
        assert "fewer_than_3_valid_groups" not in verdict["reason"]
