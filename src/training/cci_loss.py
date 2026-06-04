"""Counterfactual Calibration Invariance (CCI) v2 loss.

Generalises Counterfactual Logit Pairing (Garg AIES 2019) along two axes:

  1. **Per-group learned temperature**: pair distance is computed AFTER applying
     a per-group temperature ``T_g`` to the logits. ``T_g`` is jointly trained
     with the model via :class:`src.models.temperature_head.TemperatureHead`.
  2. **Jensen-Shannon divergence** (vs. CLP's L1 in logit space): the divergence
     is computed in *probability* space ``σ(z / T_g)``. JS saturates for
     confident-fair pairs and amplifies near the decision boundary —
     qualitatively different gradient dynamics than monotone-rescaled L1.

The 4 ablation cells (T fixed vs learned, divergence L1 vs JS) are all
realisable through the constructor; the same module class is reused with
different (``learnable_T``, ``divergence``) combinations.

Loss formula (per training batch):

    L_cci(x, y) = focal_loss(f(x), y)
                + λ · mean over (i, j) cf-pairs of D( σ(z_i / T_{g_i}),
                                                       σ(z_j / T_{g_j}) )

where ``z`` is the positive-class logit difference (logit_1 − logit_0) and
``D ∈ {l1, js}``. The mean is over valid (cf_mask=True) pairs only.

The actual focal-loss term is computed by the model wrapper, NOT by this loss.
This module only computes the second additive term.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_PROB_EPS = 1e-7


def _logit_diff(logits: torch.Tensor, positive_class_index: int = 1) -> torch.Tensor:
    """Convert ``(..., C)`` logits to the positive-class score in logit-difference space.

    For C=2 (our binary case), this is the standard ``z = logit_1 − logit_0``
    such that ``σ(z) = softmax(logits)[..., 1]``. For C > 2 we fall back to
    ``logit_pos − max_other_logit`` (one-vs-rest), but our pipeline never hits
    that branch in practice.
    """
    if logits.size(-1) == 2:
        return logits.select(-1, positive_class_index) - logits.select(
            -1, 1 - positive_class_index
        )
    # Multi-class fallback (unused in current pipeline, kept defensive)
    pos = logits.select(-1, positive_class_index)
    other_max = logits.clone()
    other_max[..., positive_class_index] = float("-inf")
    return pos - other_max.max(dim=-1).values


def _binary_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Element-wise Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).

    For two Bernoulli distributions parameterised by p and q in (0, 1):

        m  = 0.5 (p + q)
        JS(p, q) = 0.5 [ p log(p/m) + (1-p) log((1-p)/(1-m)) ]
                 + 0.5 [ q log(q/m) + (1-q) log((1-q)/(1-m)) ]

    Returns the per-element JS divergence (≥ 0, bounded by ln(2)).

    Numerical stability: cast inputs to fp32 first (fp16/bf16 resolution near
    1.0 is ~1e-3, so `_PROB_EPS = 1e-7` would clamp to exactly 1.0 and the
    log(1-p) terms would diverge); then clamp to ``[eps, 1-eps]``.
    """
    eps = _PROB_EPS
    # Defensive fp32 cast: production configs use precision: "fp32" so this is
    # a no-op, but anyone enabling fp16/bf16 autocast on a CCI/CLP config
    # would otherwise hit `log(1 - 1.0_fp16) = -inf` on saturated logits.
    p = p.float()
    q = q.float()
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    m = 0.5 * (p + q)
    # m is in [eps, 1-eps] by convexity, but clamp again to guard against
    # tiny floating-point drift that would log(0).
    m = torch.clamp(m, eps, 1.0 - eps)
    one_m_p = 1.0 - p
    one_m_q = 1.0 - q
    one_m_m = 1.0 - m
    term_p = p * (torch.log(p) - torch.log(m)) + one_m_p * (
        torch.log(one_m_p) - torch.log(one_m_m)
    )
    term_q = q * (torch.log(q) - torch.log(m)) + one_m_q * (
        torch.log(one_m_q) - torch.log(one_m_m)
    )
    return 0.5 * (term_p + term_q)


class CCIv2Loss(nn.Module):
    """Counterfactual Calibration Invariance loss with optional per-group T + JS.

    Parameters
    ----------
    temperature_head : TemperatureHead
        Module that returns a per-example temperature given its group id.
        For the fixed-T ablations, pass a non-learnable head; for learned-T,
        pass a learnable one. The head's parameters (if any) are owned by
        the caller, not by this loss.
    divergence : {"l1", "js"}, default "js"
        L1: ``|p - q|``. JS: binary Jensen-Shannon divergence on Bernoullis.
    positive_class_index : int, default 1
        Which class's logit-difference to pair on.

    Forward signature
    -----------------
    logits_orig    : (B, C) tensor of original-tweet logits
    logits_cf      : (B, K, C) tensor of counterfactual logits, padded to K_max
    cf_mask        : (B, K)    bool tensor; True where a counterfactual is valid
    source_group_ids : (B,)    LongTensor of identity-group indices per example
                                (used for both x_i and x_swap_j — we treat the
                                source group as the canonical group of the pair).
    """

    def __init__(
        self,
        temperature_head: nn.Module,
        divergence: str = "js",
        positive_class_index: int = 1,
    ):
        super().__init__()
        if divergence not in {"l1", "js"}:
            raise ValueError(f"divergence must be 'l1' or 'js'; got {divergence!r}")
        self.temperature_head = temperature_head
        self.divergence = divergence
        self.positive_class_index = positive_class_index

    def forward(
        self,
        logits_orig: torch.Tensor,
        logits_cf: torch.Tensor,
        cf_mask: torch.Tensor,
        source_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Defensive shape checks
        if logits_orig.dim() != 2:
            raise ValueError(
                f"logits_orig must be (B, C); got {tuple(logits_orig.shape)}"
            )
        if logits_cf.dim() != 3:
            raise ValueError(
                f"logits_cf must be (B, K, C); got {tuple(logits_cf.shape)}"
            )
        if cf_mask.dim() != 2 or cf_mask.dtype != torch.bool:
            raise ValueError(
                f"cf_mask must be 2-D bool; got shape={tuple(cf_mask.shape)}, dtype={cf_mask.dtype}"
            )
        if source_group_ids.dim() != 1:
            raise ValueError(
                f"source_group_ids must be 1-D; got shape {tuple(source_group_ids.shape)}"
            )
        B = logits_orig.shape[0]
        if logits_cf.shape[0] != B or cf_mask.shape[0] != B or source_group_ids.shape[0] != B:
            raise ValueError("Batch dim mismatch among inputs.")
        if logits_cf.shape[1] != cf_mask.shape[1]:
            raise ValueError(
                f"K dim mismatch: logits_cf K={logits_cf.shape[1]}, cf_mask K={cf_mask.shape[1]}"
            )

        n_valid = cf_mask.sum()
        if n_valid.item() == 0:
            return logits_orig.new_zeros(())

        # Convert to positive-class logit difference
        z_orig = _logit_diff(logits_orig, self.positive_class_index)  # (B,)
        z_cf = _logit_diff(logits_cf, self.positive_class_index)      # (B, K)

        # Per-example temperature for the source (B,)
        T = self.temperature_head(source_group_ids)  # (B,)

        # Apply temperature in logit space, then sigmoid → probability space
        p_orig = torch.sigmoid(z_orig / T)                  # (B,)
        p_cf = torch.sigmoid(z_cf / T.unsqueeze(1))         # (B, K), broadcast

        # Pair divergence
        if self.divergence == "l1":
            d = (p_orig.unsqueeze(1) - p_cf).abs()           # (B, K)
        else:  # js
            # Broadcast p_orig from (B,) to (B, K) for element-wise JS
            d = _binary_js_divergence(
                p_orig.unsqueeze(1).expand_as(p_cf), p_cf
            )

        d_masked = d * cf_mask.to(d.dtype)
        return d_masked.sum() / n_valid.to(d.dtype)
