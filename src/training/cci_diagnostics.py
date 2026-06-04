"""Diagnostics for CCI v2 — detects the base-rate-shortcut failure mode.

The deep-research review of CCI v2 (2026-05-03) flagged a specific risk: the
per-group learned temperature ``T_g`` may converge to values tightly
correlated with the inverse base rate of group ``g``. In that regime, CCI v2
is not learning calibration — it is learning a label-shift correction that
post-hoc multicalibration could achieve more directly.

We instrument the training loop to:

  * record ``T_g`` per group at every epoch end;
  * at end of training, compute Pearson correlation between ``T_g`` and (a)
    the per-group base rate and (b) ``1 − base rate``;
  * also compute variance of ``T_g`` across groups, since a learnable head that
    keeps every ``T_g`` near 1 is also non-informative.

The kill criterion (post-hoc analysis):

    |corr(T_g, 1 − base_rate)| > 0.9  AND  var(T_g) > 0.1

If both conditions are satisfied, the paper reports a clean negative result
with mechanism analysis: training-time per-group T collapses to a base-rate
shortcut, and post-hoc multicalibration is the principled approach.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.data.identity_inventory import PRIMARY_KEYWORD_GROUPS


@dataclass
class TemperatureTrajectory:
    """Records ``T_g`` across epochs for diagnostic analysis."""

    group_names: list[str] = field(default_factory=lambda: list(PRIMARY_KEYWORD_GROUPS))
    epoch: list[int] = field(default_factory=list)
    # One list of ``len(group_names)`` floats per recorded epoch
    temperatures: list[list[float]] = field(default_factory=list)

    def record(self, epoch_idx: int, T_per_group: torch.Tensor) -> None:
        if T_per_group.dim() != 1:
            raise ValueError(
                f"T_per_group must be 1-D; got shape {tuple(T_per_group.shape)}"
            )
        if T_per_group.shape[0] != len(self.group_names):
            raise ValueError(
                f"T_per_group has {T_per_group.shape[0]} entries; expected "
                f"{len(self.group_names)} (one per identity group)"
            )
        self.epoch.append(int(epoch_idx))
        self.temperatures.append([float(t) for t in T_per_group.detach().cpu().tolist()])

    def to_dict(self) -> dict:
        return {
            "group_names": self.group_names,
            "epoch": self.epoch,
            "temperatures": self.temperatures,
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TemperatureTrajectory":
        with open(path) as f:
            d = json.load(f)
        return cls(
            group_names=list(d["group_names"]),
            epoch=list(d["epoch"]),
            temperatures=[list(row) for row in d["temperatures"]],
        )


def compute_per_group_base_rates(
    labels: list[int],
    group_ids: list[int],
    num_groups: int,
) -> list[float]:
    """Empirical positive prevalence per group on the train split.

    Returns a list of length ``num_groups``. Groups with zero examples in the
    split get ``float('nan')`` — the correlation function ignores NaNs.
    """
    labels_arr = np.asarray(labels, dtype=np.float64)
    groups_arr = np.asarray(group_ids, dtype=np.int64)
    rates: list[float] = []
    for g in range(num_groups):
        mask = groups_arr == g
        if not mask.any():
            rates.append(float("nan"))
        else:
            rates.append(float(labels_arr[mask].mean()))
    return rates


def analyse_temperature_shortcut(
    trajectory: TemperatureTrajectory,
    base_rates: list[float],
    var_threshold: float = 0.1,
    corr_threshold: float = 0.9,
) -> dict:
    """Return the kill-criterion verdict for a finished CCI v2 run.

    Parameters
    ----------
    trajectory : TemperatureTrajectory
        Recorded ``T_g`` per epoch.
    base_rates : list[float]
        Per-group base rate (positive class prevalence) on the training split.
        Same ordering as :data:`PRIMARY_KEYWORD_GROUPS`.
    var_threshold : float
        Below this, ``T_g`` is too close to constant for the correlation to
        have meaning. Default ``0.1``.
    corr_threshold : float
        Absolute Pearson correlation above which we declare the shortcut.
        Default ``0.9``.

    Returns
    -------
    dict with:
        ``final_T_per_group``      — list, last-epoch ``T_g`` values
        ``base_rates``             — input
        ``var_T_final``            — variance across groups at last epoch
        ``corr_T_baserate``        — Pearson with base rate
        ``corr_T_inv_baserate``    — Pearson with (1 − base rate)
        ``shortcut_detected``      — bool, kill-criterion verdict
        ``reason``                 — human-readable explanation
    """
    if not trajectory.temperatures:
        return {
            "final_T_per_group": [],
            "base_rates": list(base_rates),
            "var_T_final": float("nan"),
            "corr_T_baserate": float("nan"),
            "corr_T_inv_baserate": float("nan"),
            "shortcut_detected": False,
            "reason": "no_trajectory_recorded",
        }

    T_final = np.asarray(trajectory.temperatures[-1], dtype=np.float64)
    base_rates_arr = np.asarray(base_rates, dtype=np.float64)
    valid = ~np.isnan(T_final) & ~np.isnan(base_rates_arr)
    if valid.sum() < 3:
        return {
            "final_T_per_group": T_final.tolist(),
            "base_rates": list(base_rates),
            "var_T_final": float(np.nanvar(T_final)),
            "corr_T_baserate": float("nan"),
            "corr_T_inv_baserate": float("nan"),
            "shortcut_detected": False,
            "reason": "fewer_than_3_valid_groups",
        }

    var_T = float(T_final[valid].var())
    if var_T < 1e-12:
        corr_br = float("nan")
        corr_inv_br = float("nan")
    else:
        corr_br = float(np.corrcoef(T_final[valid], base_rates_arr[valid])[0, 1])
        corr_inv_br = float(np.corrcoef(T_final[valid], 1.0 - base_rates_arr[valid])[0, 1])

    shortcut = (
        var_T > var_threshold
        and (
            (not np.isnan(corr_inv_br) and abs(corr_inv_br) > corr_threshold)
            or (not np.isnan(corr_br) and abs(corr_br) > corr_threshold)
        )
    )
    if shortcut:
        max_abs_corr = max(
            abs(corr_br) if not np.isnan(corr_br) else 0.0,
            abs(corr_inv_br) if not np.isnan(corr_inv_br) else 0.0,
        )
        reason = (
            f"shortcut_detected: var(T)={var_T:.3f} > {var_threshold} AND "
            f"max|corr(T, baserate or 1-baserate)|={max_abs_corr:.3f} > {corr_threshold}"
        )
    elif var_T <= var_threshold:
        reason = f"low_variance_T_g  (var={var_T:.3f} ≤ {var_threshold})"
    else:
        reason = (
            f"corr_below_threshold  "
            f"(corr_T_inv_baserate={corr_inv_br:.3f}, corr_T_baserate={corr_br:.3f})"
        )

    return {
        "final_T_per_group": T_final.tolist(),
        "base_rates": list(base_rates),
        "var_T_final": var_T,
        "corr_T_baserate": corr_br,
        "corr_T_inv_baserate": corr_inv_br,
        "shortcut_detected": bool(shortcut),
        "reason": reason,
    }
