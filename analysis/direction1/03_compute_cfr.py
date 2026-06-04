#!/usr/bin/env python3
"""Compute counterfactual flip-rate (CFR) metrics from cached logits.

Reads the per-checkpoint caches written by ``02_forward_pass.py`` and
produces ``analysis/direction1/results/per_checkpoint_metrics.parquet``
plus a paper-ready ``main_table_cfr.md`` summary.

Methods produced:

  * 8 sweep × 6 seeds = 48 rows for the trained CCI v2 / Davani methods
    (raw test-set logits → CFR directly)
  * 6 rows for ``model_c_deberta_focal_mask`` (the Model C baseline,
    raw test-set logits → CFR directly)
  * 6 rows for ``model_c_plus_ts`` (Model C with global Temperature
    Scaling fitted on dev). Includes the structural sanity check that
    CFR_hard is exactly preserved under TS.
  * 6 rows for ``model_c_plus_multicalibration`` (Model C with the
    Hébert-Johnson / Detommaso-style group-conditional patcher fitted on
    dev). Note that multicalibration uses the per-input-group correction,
    which means swap texts (mostly group "none") receive a different
    correction from originals (group "jews"/"israel"/...). This is the
    technically correct application; reviewer-attack defence is in
    REPORT.md §M2.

CPU-only; reads parquet caches under ``$SCRATCH_ROOT/cfr_cache/``.

Usage:
    python3 analysis/direction1/03_compute_cfr.py
    python3 analysis/direction1/03_compute_cfr.py --skip-posthoc   # only raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _utils import (  # noqa: E402
    BASELINE_METHODS,
    DATASET_KEYWORD_GROUPS,
    MULTIARCH_METHODS,
    SEEDS,
    SWEEP_METHODS,
    binary_js_divergence,
    cfr_cache_root,
    normalise_keyword,
    positive_class_z,
    read_manifest,
    seeds_for,
    sigmoid,
)
from src.calibration.multicalibration import (  # noqa: E402
    MulticalibrationPatcher,
    apply_global_temperature,
    fit_global_temperature,
)
from src.data.identity_inventory import (  # noqa: E402
    PRIMARY_KEYWORD_GROUPS,
    group_label_to_index,
    primary_keyword_group,
)
from src.utils.logging_utils import get_logger  # noqa: E402


logger = get_logger(__name__)


# Method label → human-readable name used in main_table.md.
METHOD_DISPLAY = {
    "davani_lambda05":            "Davani λ=0.5",
    "davani_lambda10":            "Davani λ=1.0",
    "davani_lambda20":            "Davani λ=2.0",
    "davani_lambda10_nofilt":     "Davani λ=1.0 (no filter)",
    "cci_v2_fixed_l1":            "CCI v2 (fixed T, L1)",
    "cci_v2_fixed_js":            "CCI v2 (fixed T, JS)",
    "cci_v2_learned_l1":          "CCI v2 (learned T, L1)",
    "cci_v2_learned_js":          "CCI v2 (learned T, JS)",
    "model_c_deberta_focal_mask": "Model C (focal+kw)",
    "model_c_plus_ts":            "Model C + post-hoc TS",
    "model_c_plus_multicalibration": "Model C + multicalibration",
}


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def _cache_dir(experiment: str, seed: int) -> Path:
    return cfr_cache_root() / f"{experiment}_seed{seed}"


def _load_cache(experiment: str, seed: int) -> dict:
    """Load test originals + swap logits + (Model C only) dev originals.

    Returns a dict with numpy arrays and a manifest. Raises FileNotFoundError
    if the cache is incomplete.
    """
    d = _cache_dir(experiment, seed)
    manifest = read_manifest(d)
    if manifest is None:
        raise FileNotFoundError(f"No manifest at {d / 'manifest.json'}")

    orig_path = d / "test_originals.parquet"
    if not orig_path.exists():
        raise FileNotFoundError(orig_path)
    orig_df = pd.read_parquet(orig_path)
    orig_logits = np.stack(
        [orig_df["logit_0"].to_numpy(np.float32),
         orig_df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    out = {
        "manifest": manifest,
        "orig_df": orig_df,
        "orig_logits": orig_logits,
    }

    swap_path = d / "test_swap_logits.parquet"
    if swap_path.exists():
        swap_df = pd.read_parquet(swap_path)
        swap_logits = np.stack(
            [swap_df["logit_0"].to_numpy(np.float32),
             swap_df["logit_1"].to_numpy(np.float32)],
            axis=1,
        )
    else:
        swap_df = pd.DataFrame()
        swap_logits = np.zeros((0, 2), dtype=np.float32)
    out["swap_df"] = swap_df
    out["swap_logits"] = swap_logits

    dev_path = d / "dev_originals.parquet"
    if dev_path.exists():
        dev_df = pd.read_parquet(dev_path)
        dev_logits = np.stack(
            [dev_df["logit_0"].to_numpy(np.float32),
             dev_df["logit_1"].to_numpy(np.float32)],
            axis=1,
        )
        out["dev_df"] = dev_df
        out["dev_logits"] = dev_logits
    return out


# ---------------------------------------------------------------------------
# CFR metric computation
# ---------------------------------------------------------------------------

def _compute_cfr_metrics(
    p_orig_per_swap: np.ndarray,    # (n_pairs,) probability of orig at each pair
    p_swap_per_pair: np.ndarray,    # (n_pairs,) probability of swap
    source_groups: np.ndarray,      # (n_pairs,) source-group label per pair
    decision_threshold: float = 0.5,
) -> dict:
    """Aggregate CFR_hard / CFR_soft_l1 / CFR_soft_js, both global and per-group.

    All inputs already aligned per-pair: ``p_orig_per_swap[i]`` is the
    original's positive-class probability for the SAME test example whose
    swap has probability ``p_swap_per_pair[i]``. This means originals get
    repeated as many times as they have valid swaps — pairs are the unit
    of aggregation throughout (matches how CCI v2's training loss is
    averaged).
    """
    n_pairs = len(p_orig_per_swap)
    if n_pairs == 0:
        return {
            "n_pairs_total": 0,
            "cfr_hard": float("nan"),
            "cfr_soft_l1": float("nan"),
            "cfr_soft_js": float("nan"),
            "cfr_per_group": {g: {"n_pairs": 0, "cfr_hard": None,
                                  "cfr_soft_l1": None, "cfr_soft_js": None}
                              for g in PRIMARY_KEYWORD_GROUPS},
        }

    pred_orig = (p_orig_per_swap >= decision_threshold).astype(np.int64)
    pred_swap = (p_swap_per_pair >= decision_threshold).astype(np.int64)

    flip = (pred_orig != pred_swap).astype(np.float64)
    l1 = np.abs(p_orig_per_swap - p_swap_per_pair)
    js = binary_js_divergence(p_orig_per_swap, p_swap_per_pair)

    cfr_per_group: dict[str, dict] = {}
    for g in PRIMARY_KEYWORD_GROUPS:
        mask = source_groups == g
        n_g = int(mask.sum())
        if n_g == 0:
            cfr_per_group[g] = {
                "n_pairs": 0, "cfr_hard": None,
                "cfr_soft_l1": None, "cfr_soft_js": None,
            }
        else:
            cfr_per_group[g] = {
                "n_pairs": n_g,
                "cfr_hard": float(flip[mask].mean()),
                "cfr_soft_l1": float(l1[mask].mean()),
                "cfr_soft_js": float(js[mask].mean()),
            }

    return {
        "n_pairs_total": int(n_pairs),
        "cfr_hard": float(flip.mean()),
        "cfr_soft_l1": float(l1.mean()),
        "cfr_soft_js": float(js.mean()),
        "cfr_per_group": cfr_per_group,
    }


def _classification_metrics_at_threshold(
    p_orig: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """F1+ and global accuracy on test originals at a given decision threshold."""
    y_pred = (p_orig >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = 2 * tp + fp + fn
    f1_pos = (2 * tp) / denom if denom > 0 else 0.0
    return {"f1_positive": f1_pos, "accuracy": float((y_pred == y_true).mean())}


# ---------------------------------------------------------------------------
# Per-row builders for trained methods + post-hoc baselines
# ---------------------------------------------------------------------------

def _row_for_raw_checkpoint(experiment: str, seed: int) -> dict:
    cache = _load_cache(experiment, seed)
    orig_logits = cache["orig_logits"]
    swap_logits = cache["swap_logits"]
    swap_df: pd.DataFrame = cache["swap_df"]

    p_orig = sigmoid(positive_class_z(orig_logits))
    p_swap = sigmoid(positive_class_z(swap_logits))

    if len(swap_df) == 0:
        cfr = _compute_cfr_metrics(np.array([]), np.array([]), np.array([]))
    else:
        # Align: each swap row references one test_idx; build per-pair arrays
        test_idx_per_pair = swap_df["test_idx"].to_numpy(np.int64)
        p_orig_per_swap = p_orig[test_idx_per_pair]
        cfr = _compute_cfr_metrics(
            p_orig_per_swap=p_orig_per_swap,
            p_swap_per_pair=p_swap,
            source_groups=swap_df["source_group"].astype(str).to_numpy(),
        )

    cls = _classification_metrics_at_threshold(
        p_orig, cache["orig_df"]["true_label"].to_numpy(np.int64),
    )

    return {
        "method": experiment,
        "method_kind": "trained",
        "seed": seed,
        **cls,
        **{k: v for k, v in cfr.items() if k != "cfr_per_group"},
        "cfr_per_group": json.dumps(cfr["cfr_per_group"]),
    }


def _row_for_model_c_plus_ts(seed: int, model_c_cache: dict) -> tuple[dict, dict]:
    """Compute CFR for Model C + global Temperature Scaling.

    Returns (row, sanity). ``sanity`` records whether CFR_hard is exactly
    equal to that of raw Model C — should always be True since TS is
    strictly monotone.
    """
    if "dev_logits" not in model_c_cache:
        raise RuntimeError(
            "Model C cache lacks dev_originals.parquet — TS cannot be fitted. "
            "Did 02_forward_pass.py finish successfully for the baseline?"
        )

    dev_logits = model_c_cache["dev_logits"]
    dev_labels = model_c_cache["dev_df"]["true_label"].to_numpy(np.int64)
    T = fit_global_temperature(dev_logits, dev_labels)
    logger.info("[ts seed=%d] fitted T = %.4f", seed, T)

    p_orig_ts = apply_global_temperature(model_c_cache["orig_logits"], T)
    if len(model_c_cache["swap_df"]) > 0:
        p_swap_ts = apply_global_temperature(model_c_cache["swap_logits"], T)
        test_idx_per_pair = model_c_cache["swap_df"]["test_idx"].to_numpy(np.int64)
        p_orig_per_swap_ts = p_orig_ts[test_idx_per_pair]
        cfr_ts = _compute_cfr_metrics(
            p_orig_per_swap=p_orig_per_swap_ts,
            p_swap_per_pair=p_swap_ts,
            source_groups=model_c_cache["swap_df"]["source_group"].astype(str).to_numpy(),
        )
    else:
        cfr_ts = _compute_cfr_metrics(np.array([]), np.array([]), np.array([]))

    # Sanity: CFR_hard with TS must equal CFR_hard raw (TS preserves argmax)
    p_orig_raw = sigmoid(positive_class_z(model_c_cache["orig_logits"]))
    if len(model_c_cache["swap_df"]) > 0:
        p_swap_raw = sigmoid(positive_class_z(model_c_cache["swap_logits"]))
        test_idx_per_pair = model_c_cache["swap_df"]["test_idx"].to_numpy(np.int64)
        flip_raw = (
            (p_orig_raw[test_idx_per_pair] >= 0.5).astype(int)
            != (p_swap_raw >= 0.5).astype(int)
        )
        flip_ts = (
            (p_orig_per_swap_ts >= 0.5).astype(int)
            != (p_swap_ts >= 0.5).astype(int)
        )
        cfr_hard_raw = float(flip_raw.mean())
        cfr_hard_ts = float(flip_ts.mean())
        equal = bool(np.array_equal(flip_raw, flip_ts))
    else:
        cfr_hard_raw = float("nan")
        cfr_hard_ts = float("nan")
        equal = True

    sanity = {
        "seed": seed,
        "T_fitted": float(T),
        "cfr_hard_raw": cfr_hard_raw,
        "cfr_hard_ts": cfr_hard_ts,
        "exact_match": equal,
    }
    if not equal:
        logger.warning(
            "[ts seed=%d] CFR_hard NOT preserved: raw=%.6f, ts=%.6f. "
            "This violates the structural claim — investigate.",
            seed, cfr_hard_raw, cfr_hard_ts,
        )

    cls = _classification_metrics_at_threshold(
        p_orig_ts, model_c_cache["orig_df"]["true_label"].to_numpy(np.int64),
    )

    row = {
        "method": "model_c_plus_ts",
        "method_kind": "posthoc",
        "seed": seed,
        **cls,
        **{k: v for k, v in cfr_ts.items() if k != "cfr_per_group"},
        "cfr_per_group": json.dumps(cfr_ts["cfr_per_group"]),
        "T_fitted": float(T),
    }
    return row, sanity


def _row_for_model_c_plus_multicalibration(
    seed: int, model_c_cache: dict, n_bins: int = 10,
) -> dict:
    """Compute CFR for Model C + group-conditional multicalibration patcher."""
    if "dev_logits" not in model_c_cache:
        raise RuntimeError(
            "Model C cache lacks dev_originals.parquet — multicalibration "
            "cannot be fitted. Did 02_forward_pass.py finish for the baseline?"
        )

    dev_df: pd.DataFrame = model_c_cache["dev_df"]
    dev_logits = model_c_cache["dev_logits"]
    dev_labels = dev_df["true_label"].to_numpy(np.int64)
    dev_group_ids = np.array(
        [group_label_to_index(primary_keyword_group(t))
         for t in dev_df["text"].astype(str)],
        dtype=np.int64,
    )

    patcher = MulticalibrationPatcher(
        num_groups=len(PRIMARY_KEYWORD_GROUPS),
        n_bins=n_bins,
    ).fit(dev_logits, dev_group_ids, dev_labels.astype(np.float64))

    # Apply to test originals
    orig_df: pd.DataFrame = model_c_cache["orig_df"]
    test_orig_groups = np.array(
        [group_label_to_index(primary_keyword_group(t))
         for t in orig_df["text"].astype(str)],
        dtype=np.int64,
    )
    p_orig_mc = patcher.transform_probs(model_c_cache["orig_logits"], test_orig_groups)

    # Apply to swap variants — choice B: per-input-group correction
    if len(model_c_cache["swap_df"]) > 0:
        swap_df: pd.DataFrame = model_c_cache["swap_df"]
        swap_groups_post = np.array(
            [group_label_to_index(primary_keyword_group(t))
             for t in swap_df["swap_text"].astype(str)],
            dtype=np.int64,
        )
        p_swap_mc = patcher.transform_probs(model_c_cache["swap_logits"], swap_groups_post)
        test_idx_per_pair = swap_df["test_idx"].to_numpy(np.int64)
        cfr_mc = _compute_cfr_metrics(
            p_orig_per_swap=p_orig_mc[test_idx_per_pair],
            p_swap_per_pair=p_swap_mc,
            source_groups=swap_df["source_group"].astype(str).to_numpy(),
        )
    else:
        cfr_mc = _compute_cfr_metrics(np.array([]), np.array([]), np.array([]))

    cls = _classification_metrics_at_threshold(
        p_orig_mc, orig_df["true_label"].to_numpy(np.int64),
    )

    return {
        "method": "model_c_plus_multicalibration",
        "method_kind": "posthoc",
        "seed": seed,
        **cls,
        **{k: v for k, v in cfr_mc.items() if k != "cfr_per_group"},
        "cfr_per_group": json.dumps(cfr_mc["cfr_per_group"]),
        "patcher_n_iterations": patcher._n_iterations,
        "patcher_max_residual": patcher._max_residual,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def _gather_rows(skip_posthoc: bool = False) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sanity_records: list[dict] = []

    # Trained checkpoints (sweep + Model C raw)
    for exp in SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS:
        for seed in seeds_for(exp):
            try:
                row = _row_for_raw_checkpoint(exp, seed)
                rows.append(row)
                logger.info(
                    "[raw] %s seed=%d  F1+=%.4f  CFR_hard=%.4f  CFR_soft_js=%.4f",
                    exp, seed, row["f1_positive"], row["cfr_hard"], row["cfr_soft_js"],
                )
            except FileNotFoundError as e:
                logger.warning("[raw] %s seed=%d MISSING (%s)", exp, seed, e)

    if skip_posthoc:
        return rows, sanity_records

    # Post-hoc on Model C
    for seed in SEEDS:
        try:
            mc_cache = _load_cache(BASELINE_METHODS[0], seed)
        except FileNotFoundError as e:
            logger.warning("[posthoc] Model C cache missing for seed=%d: %s", seed, e)
            continue

        row_ts, sanity = _row_for_model_c_plus_ts(seed, mc_cache)
        rows.append(row_ts)
        sanity_records.append(sanity)
        logger.info(
            "[posthoc] model_c_plus_ts seed=%d  T=%.3f  CFR_hard=%.4f  exact=%s",
            seed, sanity["T_fitted"], row_ts["cfr_hard"], sanity["exact_match"],
        )

        row_mc = _row_for_model_c_plus_multicalibration(seed, mc_cache)
        rows.append(row_mc)
        logger.info(
            "[posthoc] model_c_plus_multicalibration seed=%d  CFR_hard=%.4f  CFR_soft_js=%.4f",
            seed, row_mc["cfr_hard"], row_mc["cfr_soft_js"],
        )

    return rows, sanity_records


def _agg_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per method across the available seeds."""
    metric_cols = [
        "f1_positive", "cfr_hard", "cfr_soft_l1", "cfr_soft_js", "n_pairs_total",
    ]
    summary_rows = []
    for method, group in df.groupby("method"):
        row = {"method": method, "n_seeds": len(group),
               "method_kind": str(group["method_kind"].iloc[0])}
        for c in metric_cols:
            vals = group[c].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                row[f"{c}_mean"] = None
                row[f"{c}_std"] = None
            else:
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _build_main_table_md(summary: pd.DataFrame, output_path: Path) -> None:
    """Markdown table sorted into families (sweep, baseline, post-hoc)."""
    family_order = (
        "trained",   # sweep + Model C raw
        "posthoc",   # Model C + TS, Model C + multicalib
    )

    lines = ["# Direction-1 main table — Counterfactual Flip Rate (CFR)\n",
             "Mean ± std across seeds (n=6 unless otherwise noted).\n",
             "\n"]
    lines.append("| Method | n_seeds | F1+ | CFR_hard | CFR_soft_L1 | CFR_soft_JS | n_pairs |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    def _fmt(method_row: pd.Series, key: str) -> str:
        m = method_row.get(f"{key}_mean")
        s = method_row.get(f"{key}_std")
        if m is None or pd.isna(m):
            return "—"
        return f"{m:.4f} ± {s:.4f}"

    # Order: trained sweep first (in METHOD_DISPLAY order), then trained
    # baseline (Model C), then post-hoc.
    method_order = list(METHOD_DISPLAY.keys())
    summary_by_method = {r["method"]: r for _, r in summary.iterrows()}
    for method in method_order:
        if method not in summary_by_method:
            continue
        r = summary_by_method[method]
        display = METHOD_DISPLAY[method]
        lines.append(
            f"| {display} | {int(r['n_seeds'])} | "
            f"{_fmt(r, 'f1_positive')} | "
            f"{_fmt(r, 'cfr_hard')} | "
            f"{_fmt(r, 'cfr_soft_l1')} | "
            f"{_fmt(r, 'cfr_soft_js')} | "
            f"{int(r['n_pairs_total_mean'] or 0)} |\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute CFR per checkpoint + post-hoc baselines."
    )
    parser.add_argument(
        "--skip-posthoc",
        action="store_true",
        help="Skip Model C + TS / Model C + multicalibration variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_HERE / "results"),
        help="Where per_checkpoint_metrics.parquet etc. land.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, sanity = _gather_rows(skip_posthoc=args.skip_posthoc)
    if not rows:
        logger.error("No rows collected. Check that 02_forward_pass.py has run.")
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(output_dir / "per_checkpoint_cfr.parquet", index=False)
    logger.info("Wrote %s (%d rows)", output_dir / "per_checkpoint_cfr.parquet", len(df))

    summary = _agg_table(df)
    summary.to_csv(output_dir / "cfr_summary.csv", index=False)
    logger.info("Wrote %s", output_dir / "cfr_summary.csv")

    _build_main_table_md(summary, output_dir / "main_table_cfr.md")
    logger.info("Wrote %s", output_dir / "main_table_cfr.md")

    if sanity:
        sanity_df = pd.DataFrame(sanity)
        sanity_df.to_csv(output_dir / "ts_sanity_check.csv", index=False)
        logger.info("Wrote %s", output_dir / "ts_sanity_check.csv")
        not_exact = sanity_df[~sanity_df["exact_match"]]
        if len(not_exact) == 0:
            logger.info(
                "[SANITY ✓] CFR_hard exactly preserved by TS in 6/6 Model C seeds — "
                "the structural claim holds empirically."
            )
        else:
            logger.warning(
                "[SANITY ✗] CFR_hard NOT preserved by TS in %d seed(s) — investigate.",
                len(not_exact),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
