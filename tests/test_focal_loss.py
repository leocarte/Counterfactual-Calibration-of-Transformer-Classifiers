"""Tests for focal loss."""
import pytest
import torch
from src.models.focal_loss import FocalLoss


class TestFocalLoss:
    def test_output_shape(self):
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.shape == ()  # scalar

    def test_gamma_zero_equals_ce(self):
        """With gamma=0, focal loss should approximate cross-entropy."""
        focal = FocalLoss(gamma=0.0, alpha=0.5)
        ce = torch.nn.CrossEntropyLoss()

        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))

        focal_val = focal(logits, targets).item()
        ce_val = ce(logits, targets).item()

        # Should be close (not exact due to alpha weighting)
        assert abs(focal_val - ce_val) < 1.0

    def test_well_classified_get_lower_loss(self):
        """Focal loss should downweight well-classified examples."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.5)

        # Well-classified: high logit for correct class
        easy_logits = torch.tensor([[5.0, -5.0]])  # clearly class 0
        easy_target = torch.tensor([0])

        # Hard: low logit for correct class
        hard_logits = torch.tensor([[0.1, -0.1]])  # barely class 0
        hard_target = torch.tensor([0])

        easy_loss = loss_fn(easy_logits, easy_target).item()
        hard_loss = loss_fn(hard_logits, hard_target).item()

        assert easy_loss < hard_loss

    def test_no_nan(self):
        loss_fn = FocalLoss()
        logits = torch.zeros(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
