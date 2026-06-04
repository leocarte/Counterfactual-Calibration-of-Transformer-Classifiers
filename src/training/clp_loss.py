"""Counterfactual Logit Pairing loss (Garg et al. AIES 2019).

Penalizes the difference between the model's logit on the original tweet
``f(x)`` and on each symmetry-filtered counterfactual ``f(x_swap)``:

    L_clp(x) = mean over valid pairs of  | f(x) - f(x_swap) |

For binary classification we use the positive-class logit only — the negative
logit is its mirror under softmax, so penalising one is sufficient and avoids
double-counting.

The loss expects the trainer to have already produced counterfactual logits
via the same model. It does not run the model itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLPLoss(nn.Module):
    """L1 (or L2) logit-pairing loss between original and counterfactual examples.

    Parameters
    ----------
    divergence : {"l1", "l2"}, default "l1"
        L1 matches Garg's original formulation. L2 is offered for ablation —
        it has stronger gradient near saturation but heavier penalty on
        outliers.
    positive_class_index : int, default 1
        Which class's logit to pair on. For binary antisemitism detection the
        positive class is index 1.

    Forward signature
    -----------------
    logits_orig : (B, C) tensor of original-tweet logits
    logits_cf   : (B, K, C) tensor of counterfactual logits, padded to K_max
    cf_mask     : (B, K)    bool tensor; True where a counterfactual is valid

    Returns
    -------
    A scalar tensor: mean pair distance over valid (B, K) entries. If no
    counterfactuals are valid in the batch (e.g. no identity tokens at all),
    returns ``0.0`` (zero gradient — CLP is silent on this batch).
    """

    def __init__(self, divergence: str = "l1", positive_class_index: int = 1):
        super().__init__()
        if divergence not in {"l1", "l2"}:
            raise ValueError(f"divergence must be 'l1' or 'l2'; got {divergence!r}")
        self.divergence = divergence
        self.positive_class_index = positive_class_index

    def forward(
        self,
        logits_orig: torch.Tensor,
        logits_cf: torch.Tensor,
        cf_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Defensive shape checks — these caught real bugs in early integration.
        if logits_orig.dim() != 2:
            raise ValueError(
                f"logits_orig must be (B, C); got shape {tuple(logits_orig.shape)}"
            )
        if logits_cf.dim() != 3:
            raise ValueError(
                f"logits_cf must be (B, K, C); got shape {tuple(logits_cf.shape)}"
            )
        if cf_mask.dim() != 2 or cf_mask.dtype != torch.bool:
            raise ValueError(
                f"cf_mask must be a 2-D bool tensor; got shape={tuple(cf_mask.shape)}, "
                f"dtype={cf_mask.dtype}"
            )
        if not (logits_orig.shape[0] == logits_cf.shape[0] == cf_mask.shape[0]):
            raise ValueError(
                "Batch dim mismatch: "
                f"logits_orig={logits_orig.shape[0]}, logits_cf={logits_cf.shape[0]}, "
                f"cf_mask={cf_mask.shape[0]}"
            )
        if logits_cf.shape[1] != cf_mask.shape[1]:
            raise ValueError(
                f"K dim mismatch: logits_cf K={logits_cf.shape[1]}, cf_mask K={cf_mask.shape[1]}"
            )

        # Empty batch — return zero with correct dtype/device for autograd.
        n_valid = cf_mask.sum()
        if n_valid.item() == 0:
            return logits_orig.new_zeros(())

        # Gather positive-class logit
        f_orig = logits_orig[:, self.positive_class_index].unsqueeze(1)  # (B, 1)
        f_cf = logits_cf[:, :, self.positive_class_index]  # (B, K)

        # Pair distance — broadcast (B, 1) - (B, K) → (B, K)
        if self.divergence == "l1":
            d = (f_orig - f_cf).abs()
        else:  # l2
            d = (f_orig - f_cf) ** 2

        # Mean over valid pairs only
        d_masked = d * cf_mask.to(d.dtype)
        return d_masked.sum() / n_valid.to(d.dtype)
