"""Tests for src/training/cci_loss.py."""
import math

import pytest
import torch

from src.models.temperature_head import TemperatureHead
from src.training.cci_loss import CCIv2Loss, _binary_js_divergence


def _make_logits(z: torch.Tensor) -> torch.Tensor:
    """Build (..., 2) logits with logit_1 - logit_0 = z."""
    zeros = torch.zeros_like(z)
    return torch.stack([zeros, z], dim=-1)


# ---------------- _binary_js_divergence -----------------------------------

class TestBinaryJS:
    def test_identical_zero(self):
        p = torch.tensor([0.3, 0.7, 0.5])
        out = _binary_js_divergence(p, p)
        assert torch.allclose(out, torch.zeros_like(p), atol=1e-6)

    def test_max_at_extremes(self):
        # Maximum JS for binary distributions is ln(2) at p=0, q=1 (or vice versa)
        p = torch.tensor([1e-7])
        q = torch.tensor([1.0 - 1e-7])
        out = _binary_js_divergence(p, q)
        assert out.item() == pytest.approx(math.log(2.0), abs=1e-3)

    def test_symmetric(self):
        p = torch.tensor([0.2, 0.8])
        q = torch.tensor([0.7, 0.3])
        out_pq = _binary_js_divergence(p, q)
        out_qp = _binary_js_divergence(q, p)
        assert torch.allclose(out_pq, out_qp, atol=1e-6)

    def test_non_negative(self):
        torch.manual_seed(0)
        p = torch.rand(50)
        q = torch.rand(50)
        out = _binary_js_divergence(p, q)
        assert (out >= 0).all()


# ---------------- CCIv2Loss ------------------------------------------------

@pytest.fixture
def fixed_head():
    return TemperatureHead(num_groups=3, init_T=1.0, learnable=False)


@pytest.fixture
def learned_head():
    return TemperatureHead(num_groups=3, init_T=1.0, learnable=True)


class TestCCIv2LossBasics:
    def test_zero_pairs_returns_zero(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="js")
        logits_orig = _make_logits(torch.tensor([1.0, -2.0]))
        logits_cf = _make_logits(torch.zeros(2, 0))
        cf_mask = torch.zeros(2, 0, dtype=torch.bool)
        gids = torch.zeros(2, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        assert out.item() == 0.0

    def test_all_masked_out_returns_zero(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        logits_orig = _make_logits(torch.tensor([1.0, -2.0]))
        logits_cf = _make_logits(torch.randn(2, 3))
        cf_mask = torch.zeros(2, 3, dtype=torch.bool)
        gids = torch.zeros(2, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        assert out.item() == 0.0

    def test_identical_logits_l1_zero(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        z_orig = torch.tensor([1.0, -1.0])
        logits_orig = _make_logits(z_orig)
        # CF copies of original
        logits_cf = _make_logits(z_orig.unsqueeze(1).expand(-1, 2).clone())
        cf_mask = torch.ones(2, 2, dtype=torch.bool)
        gids = torch.zeros(2, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_identical_logits_js_zero(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="js")
        z_orig = torch.tensor([1.0, -1.0])
        logits_orig = _make_logits(z_orig)
        logits_cf = _make_logits(z_orig.unsqueeze(1).expand(-1, 2).clone())
        cf_mask = torch.ones(2, 2, dtype=torch.bool)
        gids = torch.zeros(2, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_invalid_divergence(self, fixed_head):
        with pytest.raises(ValueError):
            CCIv2Loss(fixed_head, divergence="kl")


class TestCCIv2LossL1Math:
    def test_l1_hand_computed(self, fixed_head):
        """T=1, z_orig=2.0 → p≈0.881; z_cf=0.0 → p=0.5; |0.881-0.5|≈0.381."""
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        logits_orig = _make_logits(torch.tensor([2.0]))
        logits_cf = _make_logits(torch.tensor([[0.0]]))
        cf_mask = torch.tensor([[True]])
        gids = torch.zeros(1, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        expected = abs(torch.sigmoid(torch.tensor(2.0)).item() - 0.5)
        assert out.item() == pytest.approx(expected, abs=1e-6)

    def test_temperature_changes_loss(self):
        """T=2 vs T=1 should give a smaller pair distance for same logits.

        Larger T → more uniform probabilities → smaller |p_orig - p_cf|.
        """
        head_t1 = TemperatureHead(num_groups=1, init_T=1.0, learnable=False)
        head_t2 = TemperatureHead(num_groups=1, init_T=2.0, learnable=False)
        logits_orig = _make_logits(torch.tensor([3.0]))
        logits_cf = _make_logits(torch.tensor([[0.0]]))
        cf_mask = torch.tensor([[True]])
        gids = torch.zeros(1, dtype=torch.long)
        loss_t1 = CCIv2Loss(head_t1, divergence="l1")(logits_orig, logits_cf, cf_mask, gids)
        loss_t2 = CCIv2Loss(head_t2, divergence="l1")(logits_orig, logits_cf, cf_mask, gids)
        assert loss_t2.item() < loss_t1.item()


class TestCCIv2LossGradient:
    def test_gradient_flows_to_logits(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="js")
        logits_orig = _make_logits(torch.tensor([2.0])).requires_grad_(True)
        logits_cf = _make_logits(torch.tensor([[0.0]])).requires_grad_(True)
        cf_mask = torch.tensor([[True]])
        gids = torch.zeros(1, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        out.backward()
        assert logits_orig.grad is not None
        assert logits_cf.grad is not None

    def test_gradient_flows_to_temperature_when_learnable(self, learned_head):
        loss_fn = CCIv2Loss(learned_head, divergence="js")
        logits_orig = _make_logits(torch.tensor([2.0]))
        logits_cf = _make_logits(torch.tensor([[0.0]]))
        cf_mask = torch.tensor([[True]])
        gids = torch.zeros(1, dtype=torch.long)
        out = loss_fn(logits_orig, logits_cf, cf_mask, gids)
        out.backward()
        assert learned_head.log_T.grad is not None
        # Temperature for group 0 should have non-zero gradient
        assert learned_head.log_T.grad[0].item() != 0.0

    def test_no_gradient_to_temperature_when_fixed(self, fixed_head):
        # log_T is a buffer, not a parameter — cannot accumulate grad
        param_names = [n for n, _ in fixed_head.named_parameters()]
        assert "log_T" not in param_names


class TestCCIv2LossShapeChecks:
    def test_invalid_logits_orig_dim(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4),
                torch.zeros(4, 2, 2),
                torch.zeros(4, 2, dtype=torch.bool),
                torch.zeros(4, dtype=torch.long),
            )

    def test_invalid_logits_cf_dim(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 2),
                torch.zeros(4, 2, dtype=torch.bool),
                torch.zeros(4, dtype=torch.long),
            )

    def test_invalid_cf_mask_dtype(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 2, 2),
                torch.zeros(4, 2),  # float, not bool
                torch.zeros(4, dtype=torch.long),
            )

    def test_invalid_group_dim(self, fixed_head):
        loss_fn = CCIv2Loss(fixed_head, divergence="l1")
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 2, 2),
                torch.zeros(4, 2, dtype=torch.bool),
                torch.zeros(4, 1, dtype=torch.long),  # 2-D group ids
            )
