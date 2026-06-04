"""Focal Loss for class-imbalanced classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    Reduces loss contribution from well-classified examples,
    focusing training on hard/ambiguous boundary cases.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. gamma=0 recovers CE. gamma=2 is standard.
        alpha: Class weight. Can be a scalar (applied to positive class)
               or a tensor of per-class weights.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # p_t = probability of correct class
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            alpha_t = self.alpha.to(targets.device)[targets]

        # Focal term
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Cross-entropy
        ce_loss = -torch.log(p_t + 1e-8)

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
