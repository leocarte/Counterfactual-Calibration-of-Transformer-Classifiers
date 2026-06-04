"""Shared helpers for the Direction-1 CFR / FPED-FNED / group-ECE analysis.

The functions in this module are used by every numbered script under
``analysis/direction1/``. Keep this file thin — pure utility, no orchestration.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
# torch is imported lazily inside load_model_state() — the CPU-only scripts
# (01_build_swap_manifest.py, 03_compute_cfr.py, 04_..., 05_..., 06_...,
# 07_..., 08_..., 09_...) run on the jumphost which does not have torch.
# Only 02_forward_pass.py runs inside the RCP container with torch.


# ---------------------------------------------------------------------------
# Paths (resolve based on runtime environment — container vs jumphost)
# ---------------------------------------------------------------------------

_CONTAINER_SCRATCH = Path("/scratch/case/antisemitism-results")
_JUMPHOST_SCRATCH = Path(
    "/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/case/antisemitism-results"
)


def resolve_scratch_root() -> Path:
    """Return the right scratch root for the current runtime.

    Inside the RCP container the PVC is mounted at ``/scratch/...``; on the
    jumphost the same PVC is at ``/mnt/...``. We prefer the container path
    if it exists (cheaper) and fall back to the jumphost path. Override via
    ``SCRATCH_ROOT`` env var for testing.
    """
    env = os.environ.get("SCRATCH_ROOT")
    if env:
        return Path(env)
    if _CONTAINER_SCRATCH.exists():
        return _CONTAINER_SCRATCH
    return _JUMPHOST_SCRATCH


def cfr_cache_root() -> Path:
    """Where ``01_forward_pass.py`` writes per-checkpoint logit caches."""
    return resolve_scratch_root() / "cfr_cache"


def checkpoint_dir(experiment: str, seed: int) -> Path:
    return resolve_scratch_root() / "checkpoints" / f"{experiment}_seed{seed}"


def predictions_dir(experiment: str, seed: int) -> Path:
    return resolve_scratch_root() / "predictions" / f"{experiment}_seed{seed}"


def metrics_dir(experiment: str, seed: int) -> Path:
    return resolve_scratch_root() / "metrics" / f"{experiment}_seed{seed}"


# ---------------------------------------------------------------------------
# Method enumeration
# ---------------------------------------------------------------------------

#: Trained-checkpoint methods we forward-pass for CFR. Excludes the post-hoc
#: variants (Model C + TS, Model C + multicalibration) — those are produced
#: by ``05_apply_posthoc_baselines.py`` from the Model C cache.
SWEEP_METHODS: tuple[str, ...] = (
    # Davani CLP sweep
    "davani_lambda05",
    "davani_lambda10",
    "davani_lambda20",
    "davani_lambda10_nofilt",
    # CCI v2 ablation
    "cci_v2_fixed_l1",
    "cci_v2_fixed_js",
    "cci_v2_learned_l1",
    "cci_v2_learned_js",
)

#: Pre-existing baseline trained outside this branch but co-located on
#: scratch. Model C is the post-hoc baseline anchor — we forward-pass it once
#: per seed and reuse the test logits for {raw, +TS, +multicalibration}.
BASELINE_METHODS: tuple[str, ...] = (
    "model_c_deberta_focal_mask",
)

#: Six-seed grid used everywhere.
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47)


#: Multi-architecture robustness sweep (added 2026-05-14). Two cells (Model C,
#: CCI v2 fixed_js) × two larger backbones (DeBERTa-v3-large 440M, RoBERTa-large
#: 355M). Tests whether the CCI v2 effect transfers across scale and family.
MULTIARCH_METHODS: tuple[str, ...] = (
    "model_c_deberta_v3_large",
    "model_c_roberta_large",
    "cci_v2_fixed_js_deberta_v3_large",
    "cci_v2_fixed_js_roberta_large",
)


#: Seed overrides for experiments where one seed run was dropped.
#: cci_v2_fixed_js_deberta_v3_large: seed 47 ran too slowly on a100-40g
#: (~1h45 vs 30min for other seeds) and was killed; we keep n=5 for that
#: cell. Downstream stats (Mann-Whitney, paired bootstrap on common seeds)
#: handle the missing seed without invalidating the comparison.
_SEEDS_OVERRIDE: dict[str, tuple[int, ...]] = {
    "cci_v2_fixed_js_deberta_v3_large": (42, 43, 44, 45, 46),
}


def seeds_for(experiment: str) -> tuple[int, ...]:
    """Return the seed list for a given experiment.

    Most experiments use the full 6-seed grid; a few (see _SEEDS_OVERRIDE)
    have a reduced grid because one seed run was dropped during training.
    """
    return _SEEDS_OVERRIDE.get(experiment, SEEDS)


def all_checkpoints() -> list[tuple[str, int]]:
    """Return every (experiment, seed) tuple that needs a forward-pass cache.

    Includes SWEEP, BASELINE, and MULTIARCH methods. Respects per-experiment
    seed overrides (see _SEEDS_OVERRIDE).
    """
    out: list[tuple[str, int]] = []
    for exp in SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS:
        for seed in seeds_for(exp):
            out.append((exp, seed))
    return out


def all_multiarch_checkpoints() -> list[tuple[str, int]]:
    """Return only the multi-arch (experiment, seed) tuples.

    Used by ``submit_forward_pass_multiarch.sh`` to fan out the 23 new
    forward-pass jobs without re-running the original 48 sweep jobs.
    """
    out: list[tuple[str, int]] = []
    for exp in MULTIARCH_METHODS:
        for seed in seeds_for(exp):
            out.append((exp, seed))
    return out


# ---------------------------------------------------------------------------
# Checkpoint loading — handles both flat and CCI-wrapped state dicts
# ---------------------------------------------------------------------------

def load_model_state(checkpoint_path: Path, device: str = "cpu") -> dict:
    """Load a checkpoint and return only the model's flat ``state_dict``.

    CCITrainer (commit 0ebc66a) saves checkpoints as
    ``{"model": <flat state_dict>, "temperature_head": <flat state_dict>}``,
    while every other trainer saves the flat state_dict directly. This
    helper detects the wrapper and unwraps. Same policy as the inline guard
    in ``scripts/apply_multicalibration.py`` (commit 98f85e3).

    Imports ``torch`` lazily so the CPU-only scripts can use the rest of
    this module on environments (jumphost) where torch is not installed.
    """
    import torch  # local import — see module docstring for why

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and "temperature_head" in state:
        return state["model"]
    return state


# ---------------------------------------------------------------------------
# Manifest for incremental caching
# ---------------------------------------------------------------------------

_MANIFEST_VERSION = 1


@dataclass
class CacheManifest:
    """Sidecar JSON describing a per-checkpoint logit cache.

    Re-running the forward-pass step skips any checkpoint whose existing
    manifest matches the expected ``checkpoint_sha256`` and ``swap_seed``.
    """

    version: int
    experiment: str
    seed: int
    checkpoint_path: str
    checkpoint_sha256: str
    swap_engine_seed: int
    n_test_examples: int
    n_pairs_total: int
    n_pairs_per_group: dict[str, int]
    built_at_utc: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "experiment": self.experiment,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "swap_engine_seed": self.swap_engine_seed,
            "n_test_examples": self.n_test_examples,
            "n_pairs_total": self.n_pairs_total,
            "n_pairs_per_group": self.n_pairs_per_group,
            "built_at_utc": self.built_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheManifest":
        return cls(
            version=int(d["version"]),
            experiment=str(d["experiment"]),
            seed=int(d["seed"]),
            checkpoint_path=str(d["checkpoint_path"]),
            checkpoint_sha256=str(d["checkpoint_sha256"]),
            swap_engine_seed=int(d["swap_engine_seed"]),
            n_test_examples=int(d["n_test_examples"]),
            n_pairs_total=int(d["n_pairs_total"]),
            n_pairs_per_group={str(k): int(v) for k, v in d["n_pairs_per_group"].items()},
            built_at_utc=str(d["built_at_utc"]),
        )


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def write_manifest(cache_dir: Path, manifest: CacheManifest) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path(cache_dir).open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def read_manifest(cache_dir: Path) -> CacheManifest | None:
    p = manifest_path(cache_dir)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return CacheManifest.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def file_sha256(path: Path) -> str:
    """Stream-hash a (possibly multi-GB) file. Used to detect checkpoint
    rebuilds — if the same path's sha256 changes, the cache is stale."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Group-id helpers (canonical for FPED/FNED + group-ECE)
# ---------------------------------------------------------------------------

#: The four GoldStandard2024 keyword groups (no "none" — every test sample
#: has a keyword by construction). Used by FPED/FNED and group-ECE.
DATASET_KEYWORD_GROUPS: tuple[str, ...] = ("Jews", "Israel", "Kikes", "ZioNazi")

#: Mapping from the dataset's verbatim keyword (case-as-stored) to the
#: lowercase canonical name we use everywhere else. The CSV-saved column
#: contains the dataset's casing; everything downstream uses the lowercase.
_KEYWORD_NORMALISE = {
    "Jews": "jews",
    "Israel": "israel",
    "Kikes": "kikes",
    "ZioNazi": "zionazi",
    # Defensive: accept lowercase too if anyone normalised earlier
    "jews": "jews",
    "israel": "israel",
    "kikes": "kikes",
    "zionazi": "zionazi",
}


def normalise_keyword(kw: str) -> str:
    """Return the lowercase canonical group label for a dataset keyword.

    Raises ValueError on unknown keywords so we catch label drift early.
    """
    out = _KEYWORD_NORMALISE.get(str(kw))
    if out is None:
        raise ValueError(
            f"Unknown keyword {kw!r}; expected one of {list(_KEYWORD_NORMALISE)}"
        )
    return out


# ---------------------------------------------------------------------------
# Sigmoid / JS-divergence in numpy (used by 02_compute_cfr.py)
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid for both branches of z."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg_exp = np.exp(z[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def positive_class_z(logits: np.ndarray, positive_class_index: int = 1) -> np.ndarray:
    """Convert ``(N, 2)`` logits to the positive-class score in
    logit-difference space (``z = logit_pos - logit_neg``).

    Matches the convention used by ``CCIv2Loss`` and the multicalibration
    patcher; see ``src/training/cci_loss.py`` and
    ``src/calibration/multicalibration.py``.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) logits, got {tuple(logits.shape)}")
    other = 1 - positive_class_index
    return logits[:, positive_class_index] - logits[:, other]


_PROB_EPS = 1e-7


def binary_js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Element-wise JS divergence between Bernoulli(p) and Bernoulli(q).

    Same closed form as ``src/training/cci_loss.py::_binary_js_divergence``,
    but in numpy so the metric step does not require torch.
    Returned values are non-negative and bounded by ``ln 2``.
    """
    p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    q = np.clip(q, _PROB_EPS, 1.0 - _PROB_EPS)
    m = np.clip(0.5 * (p + q), _PROB_EPS, 1.0 - _PROB_EPS)
    one_m_p, one_m_q, one_m_m = 1.0 - p, 1.0 - q, 1.0 - m
    term_p = p * (np.log(p) - np.log(m)) + one_m_p * (np.log(one_m_p) - np.log(one_m_m))
    term_q = q * (np.log(q) - np.log(m)) + one_m_q * (np.log(one_m_q) - np.log(one_m_m))
    return 0.5 * (term_p + term_q)


# ---------------------------------------------------------------------------
# Slug helpers — keep RCP job names ≤ 50 chars, lowercase, hyphenated
# ---------------------------------------------------------------------------

_RUNAI_NAME_RE = re.compile(r"[^a-z0-9-]+")


def runai_safe_name(parts: Iterable[str]) -> str:
    """Concatenate parts with '-' and sanitize for ``runai submit --name``."""
    raw = "-".join(parts)
    raw = raw.lower().replace("_", "-")
    raw = _RUNAI_NAME_RE.sub("-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-")
    return raw[:50]
