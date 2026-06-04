"""Per-group learnable temperature head for the CCI v2 loss.

A trivial nn.Module: one scalar log-temperature per identity group, parameterised
through an exponential so the temperature is always strictly positive. Joint
training with the rest of the model is the entire point — the contrast vs. CCI v1
(fixed T) is whether the temperature has informative dynamics during training.

The temperature is also clipped to ``[T_min, T_max]`` to prevent collapse / blow-up.
Defaults are conservative (0.5, 5.0); these match the CCI v2 design doc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemperatureHead(nn.Module):
    """Learns one scalar temperature per identity group, jointly with the model.

    Parameters
    ----------
    num_groups : int
        How many distinct identity groups receive their own temperature.
        For the antisemitism setup this is 5 (jews, israel, kikes, zionazi, none).
    init_T : float, default 1.0
        Starting value of every temperature. ``1.0`` = no calibration shift at
        the start; the model is free to adapt.
    T_min, T_max : float
        Hard clipping range applied at every forward pass. Prevents a degenerate
        temperature from breaking training (saturated logits → zero grad → stuck).
    learnable : bool, default True
        If False, the head returns ``init_T`` for all groups and ``log_T`` is
        registered as a buffer instead of a parameter. Used for the
        "fixed-T" ablation cells (C-fix-L1 / C-fix-JS).

    Notes
    -----
    * Uses ``log_T`` as the underlying parameter and ``T = exp(log_T)`` so that
      the temperature is always positive without explicit projection. Standard
      trick from Guo et al. ICML 2017's PyTorch reference implementation.
    * The clip ``[T_min, T_max]`` is applied to ``T``, not to ``log_T``. This
      is non-differentiable at the clip boundary; in practice the boundary is
      hit only when the optimizer is going wild, so the dropped gradient is
      not a meaningful loss of information.
    """

    def __init__(
        self,
        num_groups: int = 5,
        init_T: float = 1.0,
        T_min: float = 0.5,
        T_max: float = 5.0,
        learnable: bool = True,
    ):
        super().__init__()
        if num_groups < 1:
            raise ValueError(f"num_groups must be ≥ 1; got {num_groups}")
        if not (T_min > 0 and T_max > T_min):
            raise ValueError(
                f"Need 0 < T_min < T_max; got T_min={T_min}, T_max={T_max}"
            )
        if init_T < T_min or init_T > T_max:
            raise ValueError(
                f"init_T={init_T} must be within [T_min={T_min}, T_max={T_max}]"
            )

        self.num_groups = num_groups
        self.T_min = T_min
        self.T_max = T_max
        self.learnable = learnable

        log_T0 = torch.full((num_groups,), math.log(init_T), dtype=torch.float32)
        if learnable:
            self.log_T = nn.Parameter(log_T0)
        else:
            self.register_buffer("log_T", log_T0)

    def forward(self, group_ids: torch.Tensor) -> torch.Tensor:
        """Return ``T_g`` for each example, shape ``(N,)``.

        Parameters
        ----------
        group_ids : LongTensor of shape (N,)
            Integer group index per example, in ``[0, num_groups)``.
        """
        if group_ids.dtype not in (torch.int64, torch.long, torch.int32):
            raise ValueError(
                f"group_ids must be integer dtype; got {group_ids.dtype}"
            )
        if group_ids.dim() != 1:
            raise ValueError(
                f"group_ids must be 1-D; got shape {tuple(group_ids.shape)}"
            )
        T = torch.exp(self.log_T)  # (num_groups,)
        T = torch.clamp(T, min=self.T_min, max=self.T_max)
        return T[group_ids]

    def all_temperatures(self) -> torch.Tensor:
        """Return all current per-group temperatures as a 1-D tensor.

        Used by the diagnostic that correlates ``T_g`` with the per-group base
        rate at end-of-training (the kill-criterion check).
        """
        T = torch.exp(self.log_T)
        return torch.clamp(T, min=self.T_min, max=self.T_max)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, T_min={self.T_min}, T_max={self.T_max}, "
            f"learnable={self.learnable}"
        )
