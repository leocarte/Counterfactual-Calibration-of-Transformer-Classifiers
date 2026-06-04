"""Tests for src/training/clp_loss.py."""
import pytest
import torch

from src.training.clp_loss import CLPLoss


class TestCLPLoss:
    def test_zero_pairs_returns_zero(self):
        loss_fn = CLPLoss()
        logits_orig = torch.randn(4, 2)
        logits_cf = torch.zeros(4, 0, 2)  # K = 0
        cf_mask = torch.zeros(4, 0, dtype=torch.bool)
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert out.item() == 0.0
        assert out.shape == ()

    def test_all_masked_out_returns_zero(self):
        loss_fn = CLPLoss()
        logits_orig = torch.randn(4, 2)
        logits_cf = torch.randn(4, 3, 2)
        cf_mask = torch.zeros(4, 3, dtype=torch.bool)
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert out.item() == 0.0

    def test_identical_logits_zero_loss(self):
        """If counterfactual logits equal original, CLP loss is exactly zero."""
        loss_fn = CLPLoss()
        logits_orig = torch.tensor([[0.1, 0.5], [-1.0, 2.0]])
        # K=2, both cf identical to orig
        logits_cf = logits_orig.unsqueeze(1).expand(-1, 2, -1).clone()
        cf_mask = torch.ones(2, 2, dtype=torch.bool)
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert out.item() == 0.0

    def test_l1_distance_correct(self):
        """Hand-computed: (|2-1| + |2-3|) / 2 = 1.0."""
        loss_fn = CLPLoss(divergence="l1")
        logits_orig = torch.tensor([[0.0, 2.0]])  # positive logit = 2
        logits_cf = torch.tensor([[[0.0, 1.0], [0.0, 3.0]]])  # K=2
        cf_mask = torch.tensor([[True, True]])
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert pytest.approx(out.item(), abs=1e-6) == 1.0

    def test_l2_distance_correct(self):
        """Hand-computed: ((2-1)^2 + (2-3)^2) / 2 = 1.0."""
        loss_fn = CLPLoss(divergence="l2")
        logits_orig = torch.tensor([[0.0, 2.0]])
        logits_cf = torch.tensor([[[0.0, 1.0], [0.0, 3.0]]])
        cf_mask = torch.tensor([[True, True]])
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert pytest.approx(out.item(), abs=1e-6) == 1.0

    def test_mask_zeros_out_invalid_pairs(self):
        """Only the second cf is valid; loss = |2 - 5| = 3."""
        loss_fn = CLPLoss()
        logits_orig = torch.tensor([[0.0, 2.0]])
        logits_cf = torch.tensor([[[0.0, 100.0], [0.0, 5.0]]])
        cf_mask = torch.tensor([[False, True]])
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert pytest.approx(out.item(), abs=1e-6) == 3.0

    def test_uses_positive_class_index(self):
        """positive_class_index=0 should pair on the first column."""
        loss_fn = CLPLoss(positive_class_index=0)
        # Pair on column 0: |1.0 - 4.0| = 3.0
        logits_orig = torch.tensor([[1.0, 999.0]])
        logits_cf = torch.tensor([[[4.0, -999.0]]])
        cf_mask = torch.tensor([[True]])
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        assert pytest.approx(out.item(), abs=1e-6) == 3.0

    def test_invalid_divergence_raises(self):
        with pytest.raises(ValueError):
            CLPLoss(divergence="cosine")

    def test_invalid_shapes_raise(self):
        loss_fn = CLPLoss()
        # 1-D logits_orig
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4),
                torch.zeros(4, 2, 2),
                torch.zeros(4, 2, dtype=torch.bool),
            )
        # 2-D logits_cf (missing K dim)
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 2),
                torch.zeros(4, 2, dtype=torch.bool),
            )
        # cf_mask not bool
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 2, 2),
                torch.zeros(4, 2),
            )
        # Mismatched K
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 3, 2),
                torch.zeros(4, 2, dtype=torch.bool),
            )

    def test_gradient_flows(self):
        loss_fn = CLPLoss()
        logits_orig = torch.tensor([[0.0, 2.0]], requires_grad=True)
        logits_cf = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
        cf_mask = torch.tensor([[True]])
        out = loss_fn(logits_orig, logits_cf, cf_mask)
        out.backward()
        assert logits_orig.grad is not None
        assert logits_cf.grad is not None
        # Sign of gradient: increasing original positive logit increases |2-1|
        assert logits_orig.grad[0, 1].item() > 0
