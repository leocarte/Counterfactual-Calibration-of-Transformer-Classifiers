#!/usr/bin/env python3
"""Semi-supervised curves: F1+/CFR vs labeled fraction for CCI vs CDA vs Model C.

The one regime where CCI is structurally unique: at labeled fraction f, all three
methods share the same labeled subset, but CCI additionally enforces label-free
consistency on the unlabeled (1-f) pool's swaps. CDA cannot (it needs a label to
copy onto a swap); Model C cannot either. Prediction: CCI's F1+ advantage over
CDA GROWS as f shrinks -- it converts unlabeled counterfactual structure into
signal the others discard.

Inputs (cfr_cache):
  <base>_seed<N>                  -- fraction 1.0 (existing main runs)
  <base>_lf{0.1,0.25,0.5}_seed<N> -- the semi-sup sweep
for base in {model_c_deberta_focal_mask, cda_deberta_base, cci_v2_fixed_js}.

Outputs (results/):
  semisup.parquet   per (method, fraction, seed)
  semisup.csv       method x fraction (mean +- std)
  semisup.png       two panels: F1+ vs fraction, CFR_hard vs fraction

CPU-only. Reuses the cache loaders from 03_compute_cfr.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

_spec = importlib.util.spec_from_file_location("_three", _HERE / "03_compute_cfr.py")
_three = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_three)  # type: ignore[union-attr]
_load_cache = _three._load_cache
_compute_cfr_metrics = _three._compute_cfr_metrics
_classification_metrics_at_threshold = _three._classification_metrics_at_threshold

from _utils import cfr_cache_root, sigmoid, positive_class_z  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASES = ("model_c_deberta_focal_mask", "cda_deberta_base", "cci_v2_fixed_js")
DISPLAY = {
    "model_c_deberta_focal_mask": "Model C",
    "cda_deberta_base": "CDA",
    "cci_v2_fixed_js": "CCI",
}
FRACS = (0.1, 0.25, 0.5, 1.0)
SEEDS = (42, 43, 44)


def _experiment_name(base: str, frac: float) -> str:
    return base if frac == 1.0 else f"{base}_lf{frac:g}"


def _row_for(experiment: str, seed: int) -> dict | None:
    try:
        cache = _load_cache(experiment, seed)
    except FileNotFoundError as e:
        logger.warning("missing: %s seed=%d (%s)", experiment, seed, e)
        return None
    swap_df = cache["swap_df"]
    p_orig = sigmoid(positive_class_z(cache["orig_logits"]))
    if len(swap_df) == 0:
        cfr = _compute_cfr_metrics(np.array([]), np.array([]), np.array([]))
    else:
        p_swap = sigmoid(positive_class_z(cache["swap_logits"]))
        idx = swap_df["test_idx"].to_numpy(np.int64)
        cfr = _compute_cfr_metrics(
            p_orig_per_swap=p_orig[idx],
            p_swap_per_pair=p_swap,
            source_groups=swap_df["source_group"].astype(str).to_numpy(),
        )
    cls = _classification_metrics_at_threshold(
        p_orig, cache["orig_df"]["true_label"].to_numpy(np.int64),
    )
    return {
        "experiment": experiment, "seed": seed,
        "f1_positive": cls["f1_positive"],
        "cfr_hard": cfr["cfr_hard"],
        "cfr_soft_js": cfr["cfr_soft_js"],
    }


def _gather(fracs: tuple[float, ...], seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for base in BASES:
        for frac in fracs:
            exp = _experiment_name(base, frac)
            for seed in seeds:
                r = _row_for(exp, seed)
                if r is None:
                    continue
                r["method"] = DISPLAY[base]
                r["fraction"] = frac
                rows.append(r)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, f), g in df.groupby(["method", "fraction"]):
        out.append({
            "method": m, "fraction": f, "n_seeds": len(g),
            "f1_mean": float(g["f1_positive"].mean()),
            "f1_std": float(g["f1_positive"].std(ddof=1)) if len(g) > 1 else 0.0,
            "cfr_hard_mean": float(g["cfr_hard"].mean()),
            "cfr_hard_std": float(g["cfr_hard"].std(ddof=1)) if len(g) > 1 else 0.0,
        })
    return (pd.DataFrame(out)
              .sort_values(["method", "fraction"])
              .reset_index(drop=True))


def _plot(agg: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib missing -- skipping plot.")
        return
    colors = {"Model C": "#888888", "CDA": "#d62728", "CCI": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=140)
    for ax, (col_mean, col_std, ylabel, title) in zip(
        axes,
        [("f1_mean", "f1_std", "F1+ (positive class)", "Accuracy under label scarcity"),
         ("cfr_hard_mean", "cfr_hard_std", r"CFR$_{\rm hard}$", "Fragility under label scarcity")],
    ):
        for method in ("Model C", "CDA", "CCI"):
            sub = agg[agg["method"] == method].sort_values("fraction")
            if sub.empty:
                continue
            ax.errorbar(sub["fraction"], sub[col_mean], yerr=sub[col_std],
                        label=method, color=colors.get(method, "k"),
                        marker="o", capsize=3, linewidth=1.7)
        ax.set_xlabel("Labeled fraction of train")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted(agg["fraction"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logger.info("Wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser(description="Semi-supervised F1+/CFR vs labeled fraction.")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--output-dir", type=str, default=str(_HERE / "results"))
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = _gather(tuple(args.fractions), tuple(args.seeds))
    if df.empty:
        logger.error("No semi-sup caches found at %s. Did 02_forward_pass.py finish "
                     "on the *_lf* experiments?", cfr_cache_root())
        return 1
    df.to_parquet(out / "semisup.parquet", index=False)
    logger.info("Wrote %s (%d rows)", out / "semisup.parquet", len(df))

    agg = _aggregate(df)
    agg.to_csv(out / "semisup.csv", index=False)
    logger.info("Wrote %s", out / "semisup.csv")

    logger.info("=" * 64)
    logger.info("F1+ and CFR_hard vs labeled fraction (mean +- std across seeds)")
    for method in ("Model C", "CDA", "CCI"):
        sub = agg[agg["method"] == method].sort_values("fraction")
        if sub.empty:
            continue
        cells = "  ".join(
            f"f={f:g}: F1={f1:.3f} CFR={c:.4f} (n={int(n)})"
            for f, f1, c, n in zip(sub["fraction"], sub["f1_mean"],
                                   sub["cfr_hard_mean"], sub["n_seeds"])
        )
        logger.info("  %-8s %s", method, cells)

    # The headline test: CCI F1+ advantage over CDA at the smallest fraction.
    fmin = min(args.fractions)
    piv = agg[agg["fraction"] == fmin].set_index("method")["f1_mean"]
    if {"CCI", "CDA"}.issubset(piv.index):
        logger.info("At f=%g: CCI F1+ - CDA F1+ = %+.4f  (>0 supports the CCI edge)",
                    fmin, float(piv["CCI"] - piv["CDA"]))

    _plot(agg, out / "semisup.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
