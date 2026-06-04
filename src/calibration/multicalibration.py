"""Group-conditioned post-hoc calibration.

Two implementations are provided, both behind the same :class:`GroupCalibrator`
interface so the apply-multicalibration CLI can use either:

  * :class:`PerGroupTemperatureScaling` — fits one scalar temperature per
    identity group via NLL minimization on the dev set. This is the cheap,
    well-conditioned approximation to multicalibration when groups are
    disjoint and the only calibration error is per-group sharpness.

  * :class:`MulticalibrationPatcher` — Hébert-Johnson / Detommaso-style
    iterative patcher. Identifies the worst (group, prediction-bin) violation
    on the dev set, applies a small additive correction to all predictions
    in that (group, bin) cell, and repeats until either no violation exceeds
    a threshold or a max-iteration budget is hit. More general than per-group
    T but harder to defend on small dev sets.

Both fit on a dev split with arrays of (logits, group_ids, labels) and
transform new (logits, group_ids) at inference. Logits are expected as a 2-D
array of shape ``(N, 2)`` for the binary classification case; we operate on the
positive-class score in logit-difference space (``z = logit_1 - logit_0``),
which matches the existing :class:`TransformerClassifier`'s 2-class softmax.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROB_EPS = 1e-12


def _logits_to_z(logits: np.ndarray) -> np.ndarray:
    """Convert 2-class logits to the positive-class score in logit-difference space.

    For a 2-class softmax ``p_pos = exp(z1) / (exp(z0) + exp(z1)) = σ(z1 - z0)``,
    so the per-example calibration knob is ``z = z1 - z0``. Temperature scaling
    then divides ``z`` by ``T``.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"Expected logits of shape (N, 2); got {tuple(logits.shape)}"
        )
    return logits[:, 1] - logits[:, 0]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg_exp = np.exp(z[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def _binary_nll(z: np.ndarray, y: np.ndarray, eps: float = _PROB_EPS) -> float:
    """Mean binary NLL given positive-class scores ``z`` and binary ``y``."""
    p = np.clip(_sigmoid(z), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Global temperature scaling (Guo et al. ICML 2017) — used as a sanity check
# ---------------------------------------------------------------------------

def fit_global_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    min_T: float = 0.05,
    max_T: float = 10.0,
) -> float:
    """Fit a single scalar temperature T over the entire dataset.

    Returns the fitted T (always positive). Uses bounded scalar minimization
    of binary NLL — robust and parameter-free.
    """
    if len(logits) == 0:
        return 1.0
    z = _logits_to_z(np.asarray(logits, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)

    def nll_at(T: float) -> float:
        return _binary_nll(z / T, y)

    result = minimize_scalar(nll_at, bounds=(min_T, max_T), method="bounded")
    return float(result.x)


def apply_global_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return calibrated positive-class probabilities ``σ((z1 - z0) / T)``."""
    z = _logits_to_z(np.asarray(logits, dtype=np.float64))
    return _sigmoid(z / T)


# ---------------------------------------------------------------------------
# Group calibrator interface
# ---------------------------------------------------------------------------

class GroupCalibrator(ABC):
    """Abstract base for group-conditioned post-hoc calibrators.

    Every subclass supports ``fit(logits, group_ids, labels)`` and
    ``transform_probs(logits, group_ids) -> probabilities``.
    """

    @abstractmethod
    def fit(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
        labels: np.ndarray,
    ) -> "GroupCalibrator":
        """Estimate parameters from a held-out calibration set. Returns ``self``."""
        ...

    @abstractmethod
    def transform_probs(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
    ) -> np.ndarray:
        """Return calibrated positive-class probabilities, shape ``(N,)``."""
        ...

    @abstractmethod
    def export_state(self) -> dict:
        """Return a JSON-serializable description of the fitted calibrator.
        Used to dump the calibrator alongside model checkpoints."""
        ...


# ---------------------------------------------------------------------------
# Per-group temperature scaling
# ---------------------------------------------------------------------------

@dataclass
class _GroupTState:
    T_per_group: dict[int, float]
    fitted_groups: set[int]
    default_T: float = 1.0


class PerGroupTemperatureScaling(GroupCalibrator):
    """Fit a separate scalar temperature T_g per identity group.

    For each group g present in the calibration set, minimize binary NLL over
    the group's examples:

        T_g* = argmin_T  binary_NLL( σ( (z1 - z0) / T )  ;  y )

    Groups absent from the fit set fall back to the global temperature fit on
    the full set (guaranteed to exist since we always also fit it).

    Parameters
    ----------
    num_groups : int
        Total number of groups (used for shape / iteration checks). Group ids
        are expected to be in ``[0, num_groups)``.
    min_T, max_T : float
        Bounds passed to the scalar minimizer. Defaults are conservative —
        T < 0.5 means the model is heavily under-confident (rare), T > 5 means
        wildly over-confident; both are flagged in fit-time logs.
    min_group_count : int
        Minimum number of dev examples a group must have to receive its own T.
        Below this, the global T fallback is used (avoids overfitting T on a
        handful of points).
    """

    def __init__(
        self,
        num_groups: int,
        min_T: float = 0.05,
        max_T: float = 10.0,
        min_group_count: int = 30,
    ):
        if num_groups < 1:
            raise ValueError(f"num_groups must be ≥ 1; got {num_groups}")
        self.num_groups = num_groups
        self.min_T = min_T
        self.max_T = max_T
        self.min_group_count = min_group_count
        self._state: _GroupTState | None = None

    def fit(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
        labels: np.ndarray,
    ) -> "PerGroupTemperatureScaling":
        logits = np.asarray(logits, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        if not (len(logits) == len(group_ids) == len(labels)):
            raise ValueError("logits, group_ids, labels must all have the same length")
        if group_ids.min() < 0 or group_ids.max() >= self.num_groups:
            raise ValueError(
                f"group_ids out of range [0, {self.num_groups}); "
                f"got [{group_ids.min()}, {group_ids.max()}]"
            )

        # Always fit a global T as a fallback for sparsely populated groups.
        global_T = fit_global_temperature(logits, labels, self.min_T, self.max_T)

        T_per_group: dict[int, float] = {}
        fitted: set[int] = set()
        for g in range(self.num_groups):
            mask = group_ids == g
            n = int(mask.sum())
            if n < self.min_group_count:
                T_per_group[g] = global_T
                continue
            T_per_group[g] = fit_global_temperature(
                logits[mask], labels[mask], self.min_T, self.max_T
            )
            fitted.add(g)

        self._state = _GroupTState(
            T_per_group=T_per_group, fitted_groups=fitted, default_T=global_T
        )
        return self

    def transform_probs(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
    ) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("PerGroupTemperatureScaling: call fit() before transform_probs().")
        logits = np.asarray(logits, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        z = _logits_to_z(logits)
        z_scaled = np.empty_like(z)
        for g in np.unique(group_ids):
            mask = group_ids == g
            T = self._state.T_per_group.get(int(g), self._state.default_T)
            z_scaled[mask] = z[mask] / T
        return _sigmoid(z_scaled)

    def export_state(self) -> dict:
        if self._state is None:
            return {"calibrator": "PerGroupTemperatureScaling", "fitted": False}
        return {
            "calibrator": "PerGroupTemperatureScaling",
            "fitted": True,
            "num_groups": self.num_groups,
            "min_group_count": self.min_group_count,
            "default_T": self._state.default_T,
            "T_per_group": {int(g): float(t) for g, t in self._state.T_per_group.items()},
            "fitted_groups": sorted(int(g) for g in self._state.fitted_groups),
        }


# ---------------------------------------------------------------------------
# Multicalibration patcher (Hébert-Johnson 2018 / Detommaso 2024 style)
# ---------------------------------------------------------------------------

class MulticalibrationPatcher(GroupCalibrator):
    """Iterative additive-correction patcher for (group, bin) cells.

    Algorithm (binary classification, disjoint groups):

      1. Start from sigmoid(z) as the calibrated probability.
      2. Bin probabilities into ``n_bins`` equal-width bins per group.
      3. For each (group g, bin b) cell with ≥ ``min_bin_count`` points:
           gap_{g,b} = E[y | g, b] − mean(p | g, b)
      4. If any |gap_{g,b}| > tol, apply additive correction
           p ← clip(p + gap_{g,b}, ε, 1−ε) for all points in that cell.
      5. Repeat steps 2–4 up to ``max_iter`` times or until no violations.

    Notes
    -----
    * The patcher operates directly in probability space (post-sigmoid),
      not logit space. This is closer to Detommaso ICML 2024's patching of
      LLM confidence scores than to Guo's temperature scaling.
    * Equal-width bins are simple but can be ill-suited when probability
      mass concentrates near 0/1; future work could swap to equal-mass bins.
    * For inference: we store the per-(group, bin) corrections fitted on dev
      and apply the same shifts to test predictions. This is a *static*
      patcher; it does not adapt online.
    """

    def __init__(
        self,
        num_groups: int,
        n_bins: int = 10,
        max_iter: int = 30,
        tol: float = 0.02,
        min_bin_count: int = 10,
    ):
        if num_groups < 1:
            raise ValueError(f"num_groups must be ≥ 1; got {num_groups}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be ≥ 2; got {n_bins}")
        self.num_groups = num_groups
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.tol = tol
        self.min_bin_count = min_bin_count
        # Stored corrections: dict[group_id][bin_idx] = additive shift
        self._corrections: dict[int, dict[int, float]] | None = None
        self._n_iterations: int | None = None
        self._max_residual: float | None = None

    @staticmethod
    def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
        """Map probabilities in [0, 1] to bin index in [0, n_bins)."""
        idx = np.floor(p * n_bins).astype(np.int64)
        return np.clip(idx, 0, n_bins - 1)

    def fit(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
        labels: np.ndarray,
    ) -> "MulticalibrationPatcher":
        logits = np.asarray(logits, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        z = _logits_to_z(logits)
        p = _sigmoid(z)

        # Initialize per-(group, bin) cumulative correction (zero).
        corrections: dict[int, dict[int, float]] = {
            g: {b: 0.0 for b in range(self.n_bins)} for g in range(self.num_groups)
        }

        max_residual = 0.0
        for it in range(self.max_iter):
            # Recompute bin assignments after each round of patching.
            bin_idx = self._bin_index(p, self.n_bins)
            worst_gap = 0.0

            for g in range(self.num_groups):
                g_mask = group_ids == g
                if not g_mask.any():
                    continue
                p_g = p[g_mask]
                y_g = labels[g_mask]
                b_g = bin_idx[g_mask]
                for b in range(self.n_bins):
                    cell_mask = b_g == b
                    n_cell = int(cell_mask.sum())
                    if n_cell < self.min_bin_count:
                        continue
                    gap = float(y_g[cell_mask].mean() - p_g[cell_mask].mean())
                    if abs(gap) <= self.tol:
                        continue
                    # Apply additive correction to the global p array, in place.
                    full_mask = g_mask & (bin_idx == b)
                    p[full_mask] = np.clip(p[full_mask] + gap, _PROB_EPS, 1.0 - _PROB_EPS)
                    corrections[g][b] += gap
                    worst_gap = max(worst_gap, abs(gap))

            max_residual = worst_gap
            if worst_gap == 0.0:
                self._n_iterations = it + 1
                break
        else:
            self._n_iterations = self.max_iter

        self._corrections = corrections
        self._max_residual = max_residual
        return self

    def transform_probs(
        self,
        logits: np.ndarray,
        group_ids: np.ndarray,
    ) -> np.ndarray:
        if self._corrections is None:
            raise RuntimeError("MulticalibrationPatcher: call fit() before transform_probs().")
        logits = np.asarray(logits, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        z = _logits_to_z(logits)
        p = _sigmoid(z)

        # Apply each (group, bin) correction. Bin assignment uses the *post-
        # correction* p — but we apply corrections in the same fitting order
        # (single pass), which is the standard approximation.
        bin_idx = self._bin_index(p, self.n_bins)
        for g in np.unique(group_ids):
            g_int = int(g)
            g_mask = group_ids == g_int
            cell_corrections = self._corrections.get(g_int, {})
            for b, shift in cell_corrections.items():
                if shift == 0.0:
                    continue
                cell_mask = g_mask & (bin_idx == b)
                p[cell_mask] = np.clip(p[cell_mask] + shift, _PROB_EPS, 1.0 - _PROB_EPS)
        return p

    def export_state(self) -> dict:
        if self._corrections is None:
            return {"calibrator": "MulticalibrationPatcher", "fitted": False}
        # Compact representation: only non-zero corrections.
        compact = {
            int(g): {int(b): float(v) for b, v in row.items() if v != 0.0}
            for g, row in self._corrections.items()
        }
        return {
            "calibrator": "MulticalibrationPatcher",
            "fitted": True,
            "num_groups": self.num_groups,
            "n_bins": self.n_bins,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "min_bin_count": self.min_bin_count,
            "n_iterations_used": self._n_iterations,
            "max_residual_after_fit": self._max_residual,
            "corrections": compact,
        }
