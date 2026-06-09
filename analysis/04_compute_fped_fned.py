#!/usr/bin/env python3
"""Compute FPED / FNED (Dixon AIES 2018) per checkpoint.

Reads ``test_originals.parquet`` from each cache directory (and ``dev_originals``
for Model C, used to fit TS + multicalibration), then computes:

  * FPR_g, FNR_g per identity-keyword group g ∈ {Jews, Israel, Kikes, ZioNazi}
  * FPED = Σ_g |FPR_g − overall_FPR|   (Dixon AIES 2018, eq. 2)
  * FNED = Σ_g |FNR_g − overall_FNR|   (Dixon AIES 2018, eq. 3)

Produces:

  * ``per_checkpoint_fped_fned_dixon.parquet`` — one row per (method, seed)
  * ``fped_fned_summary.csv``            — mean ± std per method
  * ``main_table_fped.md``               — paper-ready markdown table

CPU-only. Run after ``02_forward_pass.py`` has produced caches for all 54
checkpoints. Same method set as ``03_compute_cfr.py``: 48 sweep × seed +
6 Model C raw + 6 Model C + TS + 6 Model C + multicalibration.

Usage:
    python3 analysis/04_compute_fped_fned.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _utils import (  # noqa: E402
    BASELINE_METHODS,
    DATASET_KEYWORD_GROUPS,
    MULTIARCH_METHODS,
    SEEDS,
    SWEEP_METHODS,
    cfr_cache_root,
    normalise_keyword,
    positive_class_z,
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
# Per-group rate computation
# ---------------------------------------------------------------------------

def _compute_fped_fned(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keyword_groups: np.ndarray,
    groups: tuple[str, ...] = DATASET_KEYWORD_GROUPS,
) -> dict:
    """Dixon AIES 2018 FPED / FNED on the four GoldStandard2024 keyword groups.

    The keyword column already partitions the test set into the four groups
    by construction (the dataset is keyword-sampled), so every example is in
    exactly one group. No "none" or unassigned bucket.

    Returns
    -------
    dict with keys: ``fpr_<group>``, ``fnr_<group>``, ``n_<group>``,
    ``n_pos_<group>``, ``fped``, ``fned``, ``worst_fpr_group``,
    ``worst_fnr_group``.
    """
    out: dict = {}
    fprs: list[float] = []
    fnrs: list[float] = []

    # Overall (pooled-across-groups) FPR / FNR — the Dixon reference rates.
    n_pos_total = int((y_true == 1).sum())
    n_neg_total = int((y_true == 0).sum())
    overall_fpr = (
        float(((y_pred == 1) & (y_true == 0)).sum() / n_neg_total)
        if n_neg_total > 0 else float("nan")
    )
    overall_fnr = (
        float(((y_pred == 0) & (y_true == 1)).sum() / n_pos_total)
        if n_pos_total > 0 else float("nan")
    )
    out["overall_fpr"] = overall_fpr
    out["overall_fnr"] = overall_fnr

    canonical = np.asarray([normalise_keyword(k) for k in keyword_groups])

    for g in groups:
        g_canonical = normalise_keyword(g)  # accept both casings; output lowercase
        mask = canonical == g_canonical
        n_g = int(mask.sum())
        if n_g == 0:
            out[f"fpr_{g_canonical}"] = float("nan")
            out[f"fnr_{g_canonical}"] = float("nan")
            out[f"n_{g_canonical}"] = 0
            out[f"n_pos_{g_canonical}"] = 0
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        n_pos = int((y_true_g == 1).sum())
        n_neg = int((y_true_g == 0).sum())
        # FPR — denominator is true negatives in this group
        if n_neg > 0:
            fpr = float(((y_pred_g == 1) & (y_true_g == 0)).sum() / n_neg)
        else:
            fpr = float("nan")
        # FNR — denominator is true positives
        if n_pos > 0:
            fnr = float(((y_pred_g == 0) & (y_true_g == 1)).sum() / n_pos)
        else:
            fnr = float("nan")
        out[f"fpr_{g_canonical}"] = fpr
        out[f"fnr_{g_canonical}"] = fnr
        out[f"n_{g_canonical}"] = n_g
        out[f"n_pos_{g_canonical}"] = n_pos
        if not np.isnan(fpr):
            fprs.append(fpr)
        if not np.isnan(fnr):
            fnrs.append(fnr)

    # Dixon (AIES 2018) per-keyword equality differences:
    # FPED = Σ_g |FPR_g − overall_FPR|,  FNED = Σ_g |FNR_g − overall_FNR|.
    if len(fprs) >= 2 and not np.isnan(overall_fpr):
        out["fped"] = float(np.sum(np.abs(np.asarray(fprs) - overall_fpr)))
    else:
        out["fped"] = float("nan")
    if len(fnrs) >= 2 and not np.isnan(overall_fnr):
        out["fned"] = float(np.sum(np.abs(np.asarray(fnrs) - overall_fnr)))
    else:
        out["fned"] = float("nan")
    # Keep the old range-based gap under unambiguous names so downstream
    # diagnostics can compare to the previous numbers if they want.
    out["fp_gap_range"] = (
        float(max(fprs) - min(fprs)) if len(fprs) >= 2 else float("nan")
    )
    out["fn_gap_range"] = (
        float(max(fnrs) - min(fnrs)) if len(fnrs) >= 2 else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# Per-method drivers
# ---------------------------------------------------------------------------

def _row_for_raw_checkpoint(experiment: str, seed: int) -> dict | None:
    """Load test_originals.parquet for a trained checkpoint and compute FPED/FNED."""
    cache_dir = cfr_cache_root() / f"{experiment}_seed{seed}"
    orig_path = cache_dir / "test_originals.parquet"
    if not orig_path.exists():
        logger.warning("Missing %s", orig_path)
        return None

    df = pd.read_parquet(orig_path)
    logits = np.stack(
        [df["logit_0"].to_numpy(np.float32), df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    p_pos = sigmoid(positive_class_z(logits))
    y_pred = (p_pos >= 0.5).astype(np.int64)
    y_true = df["true_label"].to_numpy(np.int64)
    keywords = df["keyword"].astype(str).to_numpy()

    metrics = _compute_fped_fned(y_true, y_pred, keywords)
    return {
        "method": experiment,
        "method_kind": "trained",
        "seed": seed,
        **metrics,
    }


def _row_for_model_c_plus_ts(seed: int) -> dict | None:
    """Fit TS on Model C dev cache, apply to test logits, compute FPED/FNED."""
    cache_dir = cfr_cache_root() / f"{BASELINE_METHODS[0]}_seed{seed}"
    test_path = cache_dir / "test_originals.parquet"
    dev_path = cache_dir / "dev_originals.parquet"
    if not (test_path.exists() and dev_path.exists()):
        logger.warning("Model C cache incomplete at %s", cache_dir)
        return None

    test_df = pd.read_parquet(test_path)
    dev_df = pd.read_parquet(dev_path)

    test_logits = np.stack(
        [test_df["logit_0"].to_numpy(np.float32), test_df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    dev_logits = np.stack(
        [dev_df["logit_0"].to_numpy(np.float32), dev_df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    dev_labels = dev_df["true_label"].to_numpy(np.int64)

    T = fit_global_temperature(dev_logits, dev_labels)
    p_pos_ts = apply_global_temperature(test_logits, T)
    y_pred_ts = (p_pos_ts >= 0.5).astype(np.int64)
    y_true = test_df["true_label"].to_numpy(np.int64)
    keywords = test_df["keyword"].astype(str).to_numpy()

    metrics = _compute_fped_fned(y_true, y_pred_ts, keywords)
    return {
        "method": "model_c_plus_ts",
        "method_kind": "posthoc",
        "seed": seed,
        "T_fitted": float(T),
        **metrics,
    }


def _row_for_model_c_plus_multicalibration(seed: int, n_bins: int = 10) -> dict | None:
    """Fit multicalibration patcher on Model C dev, apply to test logits, FPED/FNED."""
    cache_dir = cfr_cache_root() / f"{BASELINE_METHODS[0]}_seed{seed}"
    test_path = cache_dir / "test_originals.parquet"
    dev_path = cache_dir / "dev_originals.parquet"
    if not (test_path.exists() and dev_path.exists()):
        logger.warning("Model C cache incomplete at %s", cache_dir)
        return None

    test_df = pd.read_parquet(test_path)
    dev_df = pd.read_parquet(dev_path)

    test_logits = np.stack(
        [test_df["logit_0"].to_numpy(np.float32), test_df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    dev_logits = np.stack(
        [dev_df["logit_0"].to_numpy(np.float32), dev_df["logit_1"].to_numpy(np.float32)],
        axis=1,
    )
    dev_labels = dev_df["true_label"].to_numpy(np.int64).astype(np.float64)
    dev_group_ids = np.array(
        [group_label_to_index(primary_keyword_group(t)) for t in dev_df["text"].astype(str)],
        dtype=np.int64,
    )
    test_group_ids = np.array(
        [group_label_to_index(primary_keyword_group(t)) for t in test_df["text"].astype(str)],
        dtype=np.int64,
    )

    patcher = MulticalibrationPatcher(
        num_groups=len(PRIMARY_KEYWORD_GROUPS), n_bins=n_bins,
    ).fit(dev_logits, dev_group_ids, dev_labels)
    p_pos_mc = patcher.transform_probs(test_logits, test_group_ids)
    y_pred_mc = (p_pos_mc >= 0.5).astype(np.int64)
    y_true = test_df["true_label"].to_numpy(np.int64)
    keywords = test_df["keyword"].astype(str).to_numpy()

    metrics = _compute_fped_fned(y_true, y_pred_mc, keywords)
    return {
        "method": "model_c_plus_multicalibration",
        "method_kind": "posthoc",
        "seed": seed,
        "patcher_n_iterations": patcher._n_iterations,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Aggregation + table
# ---------------------------------------------------------------------------

def _gather_rows(skip_posthoc: bool = False) -> list[dict]:
    rows: list[dict] = []
    for exp in SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS:
        for seed in seeds_for(exp):
            row = _row_for_raw_checkpoint(exp, seed)
            if row is not None:
                rows.append(row)
                logger.info(
                    "[raw] %s seed=%d  FPED=%.4f  FNED=%.4f",
                    exp, seed, row["fped"], row["fned"],
                )

    if skip_posthoc:
        return rows

    for seed in SEEDS:
        row_ts = _row_for_model_c_plus_ts(seed)
        if row_ts is not None:
            rows.append(row_ts)
            logger.info(
                "[posthoc] model_c_plus_ts seed=%d  FPED=%.4f  FNED=%.4f",
                seed, row_ts["fped"], row_ts["fned"],
            )
        row_mc = _row_for_model_c_plus_multicalibration(seed)
        if row_mc is not None:
            rows.append(row_mc)
            logger.info(
                "[posthoc] model_c_plus_multicalibration seed=%d  FPED=%.4f  FNED=%.4f",
                seed, row_mc["fped"], row_mc["fned"],
            )

    return rows


def _agg_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "fped", "fned",
        *(f"fpr_{normalise_keyword(g)}" for g in DATASET_KEYWORD_GROUPS),
        *(f"fnr_{normalise_keyword(g)}" for g in DATASET_KEYWORD_GROUPS),
    ]
    rows = []
    for method, group in df.groupby("method"):
        row = {"method": method, "n_seeds": len(group),
               "method_kind": str(group["method_kind"].iloc[0])}
        for c in metric_cols:
            vals = group[c].dropna().to_numpy(np.float64)
            if len(vals) == 0:
                row[f"{c}_mean"] = None
                row[f"{c}_std"] = None
            else:
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # Per-group counts: fixed across seeds, take from first row
        for g in DATASET_KEYWORD_GROUPS:
            gn = normalise_keyword(g)
            row[f"n_{gn}"] = int(group[f"n_{gn}"].iloc[0])
            row[f"n_pos_{gn}"] = int(group[f"n_pos_{gn}"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_main_table_md(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# CFR-evaluation main table — FPED / FNED (Dixon AIES 2018)\n",
             "Mean ± std across seeds (n=6 unless otherwise noted). ",
             "Lower is better — both FPED and FNED measure the spread of group-conditional rates ",
             "across the 4 GoldStandard2024 identity-keyword groups (Jews, Israel, Kikes, ZioNazi).\n\n"]

    # Group counts disclosure (constant across seeds — read from the first row)
    if not summary.empty:
        ref = summary.iloc[0]
        lines.append("Per-group test counts: ")
        lines.append(", ".join(
            f"**{g}** {int(ref[f'n_{normalise_keyword(g)}'])} "
            f"({int(ref[f'n_pos_{normalise_keyword(g)}'])} pos)"
            for g in DATASET_KEYWORD_GROUPS
        ))
        lines.append(".\n\n")

    lines.append("| Method | n_seeds | FPED | FNED |\n")
    lines.append("|---|---|---|---|\n")

    def _fmt(r: pd.Series, key: str) -> str:
        m = r.get(f"{key}_mean")
        s = r.get(f"{key}_std")
        if m is None or pd.isna(m):
            return "—"
        return f"{m:.4f} ± {s:.4f}"

    method_order = list(METHOD_DISPLAY.keys())
    by_method = {r["method"]: r for _, r in summary.iterrows()}
    for method in method_order:
        if method not in by_method:
            continue
        r = by_method[method]
        display = METHOD_DISPLAY[method]
        lines.append(
            f"| {display} | {int(r['n_seeds'])} | "
            f"{_fmt(r, 'fped')} | "
            f"{_fmt(r, 'fned')} |\n"
        )

    # Per-group breakdown for the same methods
    lines.append("\n## Per-group FPR (false positive rate)\n\n")
    cols = [normalise_keyword(g) for g in DATASET_KEYWORD_GROUPS]
    lines.append("| Method | " + " | ".join(g.title() for g in cols) + " |\n")
    lines.append("|" + "---|" * (len(cols) + 1) + "\n")
    for method in method_order:
        if method not in by_method:
            continue
        r = by_method[method]
        cells = [METHOD_DISPLAY[method]]
        for g in cols:
            cells.append(_fmt(r, f"fpr_{g}"))
        lines.append("| " + " | ".join(cells) + " |\n")

    lines.append("\n## Per-group FNR (false negative rate)\n\n")
    lines.append("| Method | " + " | ".join(g.title() for g in cols) + " |\n")
    lines.append("|" + "---|" * (len(cols) + 1) + "\n")
    for method in method_order:
        if method not in by_method:
            continue
        r = by_method[method]
        cells = [METHOD_DISPLAY[method]]
        for g in cols:
            cells.append(_fmt(r, f"fnr_{g}"))
        lines.append("| " + " | ".join(cells) + " |\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute FPED / FNED per checkpoint.")
    parser.add_argument("--skip-posthoc", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(_HERE / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _gather_rows(skip_posthoc=args.skip_posthoc)
    if not rows:
        logger.error("No rows collected.")
        return 1

    df = pd.DataFrame(rows)
    out_path = output_dir / "per_checkpoint_fped_fned_dixon.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))

    summary = _agg_table(df)
    summary.to_csv(output_dir / "fped_fned_summary.csv", index=False)
    logger.info("Wrote %s", output_dir / "fped_fned_summary.csv")

    _build_main_table_md(summary, output_dir / "main_table_fped.md")
    logger.info("Wrote %s", output_dir / "main_table_fped.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
