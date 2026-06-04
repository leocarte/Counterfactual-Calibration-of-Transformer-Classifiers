"""Tests for src/models/temperature_head.py."""
import math

import pytest
import torch

from src.models.temperature_head import TemperatureHead


class TestTemperatureHead:
    def test_init_default(self):
        head = TemperatureHead(num_groups=5)
        T_all = head.all_temperatures()
        assert T_all.shape == (5,)
        # Every group starts at T = 1.0
        assert torch.allclose(T_all, torch.ones(5), atol=1e-6)

    def test_forward_shape(self):
        head = TemperatureHead(num_groups=3)
        ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        T = head(ids)
        assert T.shape == (5,)
        assert torch.all(T > 0)

    def test_forward_returns_per_group_T(self):
        head = TemperatureHead(num_groups=3)
        # Manually set distinct T per group
        with torch.no_grad():
            head.log_T.data.copy_(torch.tensor([math.log(0.5), math.log(2.0), math.log(1.0)]))
        ids = torch.tensor([0, 1, 2], dtype=torch.long)
        T = head(ids)
        assert torch.allclose(T, torch.tensor([0.5, 2.0, 1.0]), atol=1e-5)

    def test_clip_T_max(self):
        head = TemperatureHead(num_groups=2, T_max=3.0)
        # Force T well above the cap
        with torch.no_grad():
            head.log_T.data.copy_(torch.tensor([math.log(100.0), math.log(1.0)]))
        ids = torch.tensor([0, 1], dtype=torch.long)
        T = head(ids)
        assert T[0].item() == pytest.approx(3.0, abs=1e-5)
        assert T[1].item() == pytest.approx(1.0, abs=1e-5)

    def test_clip_T_min(self):
        head = TemperatureHead(num_groups=2, T_min=0.5)
        with torch.no_grad():
            head.log_T.data.copy_(torch.tensor([math.log(0.01), math.log(1.0)]))
        ids = torch.tensor([0, 1], dtype=torch.long)
        T = head(ids)
        assert T[0].item() == pytest.approx(0.5, abs=1e-5)

    def test_learnable_true_creates_parameter(self):
        head = TemperatureHead(num_groups=4, learnable=True)
        param_names = [n for n, _ in head.named_parameters()]
        assert "log_T" in param_names

    def test_learnable_false_creates_buffer(self):
        head = TemperatureHead(num_groups=4, learnable=False)
        param_names = [n for n, _ in head.named_parameters()]
        assert "log_T" not in param_names
        buffer_names = [n for n, _ in head.named_buffers()]
        assert "log_T" in buffer_names

    def test_gradient_flows_when_learnable(self):
        head = TemperatureHead(num_groups=3, learnable=True)
        ids = torch.tensor([0, 1, 2], dtype=torch.long)
        T = head(ids)
        loss = T.sum()
        loss.backward()
        assert head.log_T.grad is not None
        assert (head.log_T.grad != 0).all()

    def test_invalid_num_groups(self):
        with pytest.raises(ValueError):
            TemperatureHead(num_groups=0)

    def test_invalid_T_bounds(self):
        with pytest.raises(ValueError):
            TemperatureHead(num_groups=2, T_min=2.0, T_max=1.0)
        with pytest.raises(ValueError):
            TemperatureHead(num_groups=2, T_min=-0.5, T_max=2.0)
        with pytest.raises(ValueError):
            TemperatureHead(num_groups=2, init_T=10.0, T_max=5.0)

    def test_non_integer_group_ids_raises(self):
        head = TemperatureHead(num_groups=3)
        with pytest.raises(ValueError):
            head(torch.tensor([0.0, 1.0]))  # float, not int
