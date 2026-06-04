#!/usr/bin/env python3
"""Assemble the unified Direction-1 main table (F1+ × ECE × CFR × FPED × FNED).

Reads the three per-checkpoint metric parquets (from scripts 03, 04, 05) plus
the significance results from script 06, and emits a single paper-ready
markdown table:

    analysis/direction1/results/main_table.md

Rows = methods, columns = headline metrics. Cells show mean ± std across
seeds with a Bonferroni-significance marker (vs the registered pivot:
``cci_v2_learned_js``).

CPU-only, fast (<5 sec).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

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
    # Multi-architecture sweep (2026-05-14): tests transfer of CCI v2 to larger
    # / cross-family backbones. Without these keys the rows are silently
    # dropped by the table builder, which only keeps methods in METHOD_DISPLAY.
    "model_c_deberta_v3_large":             "Model C (DeBERTa-v3-large)",
    "model_c_roberta_large":                "Model C (RoBERTa-large)",
    "cci_v2_fixed_js_deberta_v3_large":     "CCI v2 fixed_js (DeBERTa-v3-large)",
    "cci_v2_fixed_js_roberta_large":        "CCI v2 fixed_js (RoBERTa-large)",
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())


def _agg(df: pd.DataFrame, metric: str) -> dict[str, tuple[float, float]]:
    """Return {method: (mean, std)} aggregated across seeds for ``metric``."""
    out: dict[str, tuple[float, float]] = {}
    for method, sub in df.groupby("method"):
        vals = sub[metric].dropna().to_numpy(np.float64)
        if len(vals) == 0:
            out[method] = (float("nan"), float("nan"))
        else:
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            out[method] = (mean, std)
    return out


def _fmt(stats: dict[str, tuple[float, float]], method: str, sig_marker: str = "") -> str:
    if method not in stats:
        return "—"
    m, s = stats[method]
    if np.isnan(m):
        return "—"
    return f"{m:.4f} ± {s:.4f}{sig_marker}"


def main():
    parser = argparse.ArgumentParser(description="Assemble Direction-1 main table.")
    parser.add_argument("--results-dir", type=str, default=str(_HERE / "results"))
    parser.add_argument(
        "--pivot", type=str, default="cci_v2_fixed_js",
        help="Pivot method for significance stars in the table. Changed default "
             "from cci_v2_learned_js to cci_v2_fixed_js on 2026-05-14 (Audit 2 "
             "recommendation: learned-T does not improve CFR significantly, and "
             "fixed-T has higher F1+ and lower FNED).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load three per-checkpoint metric parquets. FPED/FNED come from the
    # Dixon AIES 2018 file (sum-of-deviations, corrected 2026-05-18); the
    # legacy per_checkpoint_fped_fned.parquet reported range and is kept on
    # disk only for archaeology.
    cfr_df = pd.read_parquet(results_dir / "per_checkpoint_cfr.parquet")
    fped_df = pd.read_parquet(results_dir / "per_checkpoint_fped_fned_dixon.parquet")
    ece_df = pd.read_parquet(results_dir / "per_checkpoint_group_ece.parquet")

    # Aggregate
    f1_stats = _agg(cfr_df, "f1_positive")
    cfr_hard_stats = _agg(cfr_df, "cfr_hard")
    cfr_soft_l1_stats = _agg(cfr_df, "cfr_soft_l1")
    cfr_soft_js_stats = _agg(cfr_df, "cfr_soft_js")
    fped_stats = _agg(fped_df, "fped")
    fned_stats = _agg(fped_df, "fned")
    ece_global_stats = _agg(ece_df, "ece_global")
    ece_worst_stats = _agg(ece_df, "ece_worst_group")

    # Significance markers
    sig_path = results_dir / "significance_tests.csv"
    sig_marker_per_pair: dict[tuple[str, str], str] = {}
    if sig_path.exists():
        sig_df = pd.read_csv(sig_path)
        sig_df = sig_df[sig_df["pivot"] == args.pivot]
        for _, r in sig_df.iterrows():
            if r.get("rejects_at_bonferroni_05"):
                sig_marker_per_pair[(r["comparator"], r["metric"])] = "*"
            else:
                sig_marker_per_pair[(r["comparator"], r["metric"])] = ""
    else:
        logger.warning(
            "significance_tests.csv not found; main table will lack significance markers."
        )

    def _stats_with_sig(method: str, stats: dict, metric: str) -> str:
        if method == args.pivot:
            return _fmt(stats, method, "")
        marker = sig_marker_per_pair.get((method, metric), "")
        return _fmt(stats, method, marker)

    # alpha = 0.05 / (main-family count). Holm is per-family, so mixing the
    # 6 transfer tests into the denominator would misreport the threshold.
    sig_csv = Path(args.results_dir) / "significance_tests.csv"
    if sig_csv.exists():
        sig_df = pd.read_csv(sig_csv)
        if "family" in sig_df.columns:
            n_tests = int((sig_df["family"] == "main").sum())
        else:
            n_tests = len(sig_df)  # legacy single-family CSV
    else:
        n_tests = 54
    alpha_bonf = 0.05 / max(n_tests, 1)

    # Build table
    lines = ["# Direction-1 — Main results table\n",
             "\n",
             "Mean ± std across 6 seeds. **\\*** denotes significance vs the pivot ",
             f"(*{METHOD_DISPLAY.get(args.pivot, args.pivot)}*) at Bonferroni-corrected ",
             f"α = 0.05/{n_tests} ≈ {alpha_bonf:.4g}. ",
             "Lower is better for ECE, CFR, FPED, FNED. Higher is better for F1+.\n\n"]
    lines.append(
        "| Method | F1+ | ECE | ECE worst | CFR hard | CFR soft (L1) | "
        "CFR soft (JS) | FPED | FNED |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|\n")

    for method in METHOD_ORDER:
        if method not in f1_stats:
            continue
        display = METHOD_DISPLAY[method]
        if method == args.pivot:
            display = f"**{display}** (pivot)"
        cells = [
            display,
            _stats_with_sig(method, f1_stats, "f1_positive"),
            _stats_with_sig(method, ece_global_stats, "ece_global"),
            _stats_with_sig(method, ece_worst_stats, "ece_worst_group"),
            _stats_with_sig(method, cfr_hard_stats, "cfr_hard"),
            _stats_with_sig(method, cfr_soft_l1_stats, "cfr_soft_l1"),
            _stats_with_sig(method, cfr_soft_js_stats, "cfr_soft_js"),
            _stats_with_sig(method, fped_stats, "fped"),
            _stats_with_sig(method, fned_stats, "fned"),
        ]
        lines.append("| " + " | ".join(cells) + " |\n")

    out_path = results_dir / "main_table.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
