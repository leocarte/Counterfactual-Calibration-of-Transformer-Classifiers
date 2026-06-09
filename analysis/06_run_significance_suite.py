#!/usr/bin/env python3
"""Bonferroni-corrected paired bootstrap suite for CFR-evaluation.

Pre-registered design (paper §4):

  * **Pivot method**: ``cci_v2_learned_js`` (configurable via --pivot)
  * **Comparator set** (5 methods):
      - cci_v2_fixed_js              (isolates contribution of learned T)
      - davani_lambda05              (best Davani CLP variant)
      - model_c_deberta_focal_mask   (vanilla baseline; structural anchor)
      - model_c_plus_ts              (post-hoc TS — sanity: CFR_hard exact)
      - model_c_plus_multicalibration (post-hoc group-conditional)
  * **Metrics** (6): cfr_hard, cfr_soft_l1, cfr_soft_js, fped, fned,
    ece_worst_group
  * **Total tests**: 5 × 6 = 30
  * **Correction**: Bonferroni at α = 0.05/30 ≈ 0.001667

The bootstrap resamples the 6 seeds with replacement (Dror et al. ACL 2018
paired-over-seeds variant), n=10,000 by default.

Inputs (must already exist):
  * ``analysis/results/per_checkpoint_cfr.parquet``
  * ``analysis/results/per_checkpoint_fped_fned_dixon.parquet``
    (Dixon AIES 2018 sum-of-deviations; the legacy
    ``per_checkpoint_fped_fned.parquet`` was range-based and is preserved
    on disk only for archaeology.)
  * ``analysis/results/per_checkpoint_group_ece.parquet``

Outputs:
  * ``analysis/results/significance_tests.csv``  (one row per test)
  * ``analysis/results/significance_summary.md`` (paper-ready)

Usage:
    python3 analysis/06_run_significance_suite.py
    python3 analysis/06_run_significance_suite.py --pivot cci_v2_fixed_js
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

from _utils import SEEDS  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


METRICS: tuple[str, ...] = (
    "cfr_hard",
    "cfr_soft_l1",
    "cfr_soft_js",
    "fped",
    "fned",
    "ece_worst_group",
)


METRIC_SOURCE: dict[str, str] = {
    "cfr_hard": "per_checkpoint_cfr.parquet",
    "cfr_soft_l1": "per_checkpoint_cfr.parquet",
    "cfr_soft_js": "per_checkpoint_cfr.parquet",
    # Dixon AIES 2018 sum-of-deviations (corrected 2026-05-18).
    # The legacy file (per_checkpoint_fped_fned.parquet) reported range
    # (max - min) and is no longer consumed.
    "fped": "per_checkpoint_fped_fned_dixon.parquet",
    "fned": "per_checkpoint_fped_fned_dixon.parquet",
    "ece_worst_group": "per_checkpoint_group_ece.parquet",
}


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
    # Multi-architecture sweep (2026-05-14)
    "model_c_deberta_v3_large":             "Model C (DeBERTa-v3-large)",
    "model_c_roberta_large":                "Model C (RoBERTa-large)",
    "cci_v2_fixed_js_deberta_v3_large":     "CCI v2 fixed_js (DeBERTa-v3-large)",
    "cci_v2_fixed_js_roberta_large":        "CCI v2 fixed_js (RoBERTa-large)",
}


# ---------------------------------------------------------------------------
# Per-seed paired bootstrap on scalar metrics
# ---------------------------------------------------------------------------

def per_seed_paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 12345,
) -> dict:
    r"""Two-sided paired bootstrap + exact sign test on per-seed scalar metric values.

    ``values_a[i]`` and ``values_b[i]`` are method A's and method B's metric
    on seed i. We resample seed indices with replacement and recompute the
    mean difference for the bootstrap p-value. We also report the exact
    sign-test p-value (Audit 2 recommendation: with n=6 seeds the bootstrap
    distribution has only ~46k unique resamples and the bootstrap p-value
    can be artificially small when all seeds agree on direction; the sign
    test gives the minimum achievable p-value under H0 on a paired
    direction-only test, $p \ge 2 \cdot (1/2)^n$).

    Returns observed delta, CI, bootstrap p-value, sign-test p-value, and
    the seed-pair direction count.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"length mismatch: |A|={len(values_a)} vs |B|={len(values_b)}"
        )
    n = len(values_a)
    if n == 0:
        return {
            "metric_a": float("nan"), "metric_b": float("nan"),
            "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "p_value": float("nan"), "p_sign": float("nan"),
            "n_seeds": 0, "seed_a_lt_b": 0,
        }

    obs_a = float(np.nanmean(values_a))
    obs_b = float(np.nanmean(values_b))
    obs_delta = obs_a - obs_b

    rng = np.random.default_rng(rng_seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[k] = np.nanmean(values_a[idx]) - np.nanmean(values_b[idx])

    alpha = 1.0 - confidence_level
    ci_low = float(np.nanpercentile(deltas, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2)))

    if obs_delta >= 0:
        p_one = float(np.mean(deltas <= 0))
    else:
        p_one = float(np.mean(deltas >= 0))
    p_value = min(1.0, max(1.0 / n_resamples, 2 * p_one))

    # Exact two-sided sign test on n_eff = n_lt + n_gt (ties drop out of binomial).
    # Using 2**n with ties would be anti-conservative.
    from math import comb
    n_lt = int(np.sum(values_a < values_b))
    n_gt = int(np.sum(values_a > values_b))
    n_eff = n_lt + n_gt
    if n_eff == 0:
        p_sign = 1.0
    else:
        k_eff = min(n_lt, n_gt)
        cdf_tail = sum(comb(n_eff, i) for i in range(k_eff + 1)) / (2 ** n_eff)
        p_sign = min(1.0, 2 * cdf_tail)

    return {
        "metric_a": obs_a,
        "metric_b": obs_b,
        "delta": float(obs_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "p_sign": float(p_sign),
        "n_seeds": int(n),
        "seed_a_lt_b": n_lt,
    }


def holm_bonferroni_reject(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down procedure for family-wise error control.

    Given a vector of p-values for K tests, returns a boolean vector of
    rejections under FWER ≤ alpha. Uniformly more powerful than naive
    Bonferroni (which uses alpha/K for every test) while controlling the
    same family-wise error rate.

    Procedure:
      1. Sort p-values ascending: p_(1) ≤ ... ≤ p_(K).
      2. Reject p_(i) iff p_(j) ≤ alpha/(K - j + 1) for all j ≤ i.
    """
    K = len(p_values)
    if K == 0:
        return []
    order = np.argsort(p_values)
    p_sorted = np.asarray(p_values, dtype=np.float64)[order]
    rejects_sorted = np.zeros(K, dtype=bool)
    for i in range(K):
        threshold = alpha / (K - i)
        if p_sorted[i] <= threshold:
            rejects_sorted[i] = True
        else:
            break  # Once we fail to reject, all larger p-values also fail
    # Map back to original order
    rejects = np.zeros(K, dtype=bool)
    rejects[order] = rejects_sorted
    return rejects.tolist()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _per_seed_values(
    df: pd.DataFrame, method: str, metric: str,
) -> np.ndarray:
    """Return seed-aligned values for ``method`` on ``metric`` (length = #seeds)."""
    sub = df[df["method"] == method].sort_values("seed")
    if metric not in sub.columns:
        raise KeyError(f"metric {metric!r} not in dataframe columns: {list(sub.columns)}")
    return sub[metric].to_numpy(np.float64)


def _per_seed_aligned(
    df: pd.DataFrame, method_a: str, method_b: str, metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (vals_a, vals_b, common_seeds) restricted to seeds present in both methods.

    Handles the cci_v2_fixed_js_deberta_v3_large case where one method has
    n=5 seeds (s47 was dropped during training) and the pivot has n=6:
    we use the intersection (n=5) for that paired comparison, which is the
    standard paired-bootstrap treatment of unbalanced seed grids.
    """
    sub_a = df[df["method"] == method_a].sort_values("seed")
    sub_b = df[df["method"] == method_b].sort_values("seed")
    if metric not in sub_a.columns:
        raise KeyError(f"metric {metric!r} not in dataframe columns: {list(sub_a.columns)}")
    seeds_a = set(sub_a["seed"].astype(int).tolist())
    seeds_b = set(sub_b["seed"].astype(int).tolist())
    common = sorted(seeds_a & seeds_b)
    if not common:
        raise ValueError(
            f"No common seeds between {method_a!r} (seeds {sorted(seeds_a)}) "
            f"and {method_b!r} (seeds {sorted(seeds_b)})"
        )
    vals_a = sub_a[sub_a["seed"].astype(int).isin(common)].sort_values("seed")[metric].to_numpy(np.float64)
    vals_b = sub_b[sub_b["seed"].astype(int).isin(common)].sort_values("seed")[metric].to_numpy(np.float64)
    return vals_a, vals_b, common


def _load_metric_data(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three per-checkpoint metric parquets keyed by source filename."""
    cache: dict[str, pd.DataFrame] = {}
    for src in set(METRIC_SOURCE.values()):
        p = results_dir / src
        if not p.exists():
            raise FileNotFoundError(
                f"Required input missing: {p}. "
                f"Run 03/04/05_compute_*.py before this script."
            )
        cache[src] = pd.read_parquet(p)
    return cache


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap with Bonferroni.")
    parser.add_argument(
        "--pivot",
        type=str,
        default="cci_v2_fixed_js",
        help="The pivot (A) method against which all comparators (B) are tested. "
             "Default changed from cci_v2_learned_js to cci_v2_fixed_js on 2026-05-14 "
             "after the within-cell ablation showed learned-T is statistically "
             "indistinguishable from fixed-T on CFR_hard (p=0.25), while fixed-T "
             "has higher F1+ (0.677 vs 0.662) and lower FNED (0.543 vs 0.605).",
    )
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=[
            # Base architecture (DeBERTa-v3-base) — confirmatory comparators
            "cci_v2_learned_js",  # ablation: does learned T help vs fixed T?
            "davani_lambda05",
            "model_c_deberta_focal_mask",
            "model_c_plus_ts",
            "model_c_plus_multicalibration",
            # Multi-architecture sweep (2026-05-14) — descriptive cross-arch
            # comparisons. NOTE: these test BASE pivot vs LARGE comparator;
            # for direct within-architecture transfer tests (LARGE CCI vs
            # LARGE Model C), use --extra-pairs.
            "model_c_deberta_v3_large",
            "model_c_roberta_large",
            "cci_v2_fixed_js_deberta_v3_large",
            "cci_v2_fixed_js_roberta_large",
        ],
    )
    parser.add_argument(
        "--extra-pairs",
        type=str,
        default=(
            # Within-architecture transfer tests (Audit 2 recommendation 2026-05-14):
            # the default comparators above all use BASE pivot, so the cross-arch
            # rows test 'base vs large', NOT 'large CCI vs large Model C'.
            # These extra pairs run the direct within-arch comparisons.
            "cci_v2_fixed_js_deberta_v3_large:model_c_deberta_v3_large,"
            "cci_v2_fixed_js_roberta_large:model_c_roberta_large"
        ),
        help="Comma-separated 'pivot:comparator' pairs (each pair gets its own bootstrap "
             "on the 3 CFR metrics only). For within-architecture transfer tests where "
             "the pivot is the large CCI v2 cell instead of the base cell.",
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=12345)
    parser.add_argument("--results-dir", type=str, default=str(_HERE / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metric_data = _load_metric_data(results_dir)

    # Bonferroni: total tests = comparators × metrics
    n_tests = len(args.comparators) * len(METRICS)
    alpha_corrected = 0.05 / n_tests
    logger.info(
        "Significance suite: pivot=%s, %d comparators, %d metrics, "
        "%d total tests, Bonferroni alpha = %.6f",
        args.pivot, len(args.comparators), len(METRICS), n_tests, alpha_corrected,
    )

    rows = []
    # ----- Main family: pivot vs comparators on all metrics -----
    for comparator in args.comparators:
        for metric in METRICS:
            df_metric = metric_data[METRIC_SOURCE[metric]]
            try:
                vals_a, vals_b, common_seeds = _per_seed_aligned(
                    df_metric, args.pivot, comparator, metric,
                )
            except (KeyError, ValueError) as e:
                logger.error("Missing data for (%s, %s): %s", comparator, metric, e)
                continue

            res = per_seed_paired_bootstrap(
                vals_a, vals_b,
                n_resamples=args.n_resamples,
                rng_seed=args.rng_seed,
            )
            row = {
                "family": "main",
                "pivot": args.pivot,
                "comparator": comparator,
                "metric": metric,
                "n_common_seeds": len(common_seeds),
                **res,
                "p_bonferroni": min(1.0, res["p_value"] * n_tests),
                "rejects_at_bonferroni_05": (
                    res["p_value"] < alpha_corrected
                ),
            }
            rows.append(row)

    # ----- Extra family: within-architecture transfer pairs on CFR metrics only -----
    cfr_metrics = [m for m in METRICS if m.startswith("cfr_")]
    extra_pair_strings = [p.strip() for p in args.extra_pairs.split(",") if p.strip()]
    extra_pairs: list[tuple[str, str]] = []
    for p in extra_pair_strings:
        if ":" not in p:
            logger.warning("Skipping malformed extra-pair %r (expected 'pivot:comparator')", p)
            continue
        a, b = p.split(":", 1)
        extra_pairs.append((a.strip(), b.strip()))

    n_extra_tests = len(extra_pairs) * len(cfr_metrics)
    if n_extra_tests > 0:
        # Within-architecture transfer is a separate hypothesis family ⇒ separate
        # Bonferroni denominator. This is statistically more principled than
        # lumping into the omnibus suite (Audit 2 recommendation: pre-specify
        # hypothesis families).
        alpha_extra = 0.05 / n_extra_tests
        logger.info(
            "Extra family (within-arch transfer): %d pairs × %d CFR metrics = %d tests, "
            "Bonferroni alpha = %.6f",
            len(extra_pairs), len(cfr_metrics), n_extra_tests, alpha_extra,
        )
        for pivot_x, comparator_x in extra_pairs:
            for metric in cfr_metrics:
                df_metric = metric_data[METRIC_SOURCE[metric]]
                try:
                    vals_a, vals_b, common_seeds = _per_seed_aligned(
                        df_metric, pivot_x, comparator_x, metric,
                    )
                except (KeyError, ValueError) as e:
                    logger.error("Missing data for extra (%s vs %s, %s): %s",
                                 pivot_x, comparator_x, metric, e)
                    continue

                res = per_seed_paired_bootstrap(
                    vals_a, vals_b,
                    n_resamples=args.n_resamples,
                    rng_seed=args.rng_seed,
                )
                row = {
                    "family": "within_arch_transfer",
                    "pivot": pivot_x,
                    "comparator": comparator_x,
                    "metric": metric,
                    "n_common_seeds": len(common_seeds),
                    **res,
                    "p_bonferroni": min(1.0, res["p_value"] * n_extra_tests),
                    "rejects_at_bonferroni_05": (
                        res["p_value"] < alpha_extra
                    ),
                }
                rows.append(row)

    # ----- Apply Holm-Bonferroni separately within each family -----
    # Holm controls family-wise error at alpha=0.05 uniformly more powerfully
    # than naive Bonferroni. We apply it per family because the families test
    # distinct hypotheses (confirmatory CFR vs transfer vs Pareto audit).
    out_df = pd.DataFrame(rows)
    out_df["rejects_at_holm_bonferroni_05"] = False
    for family_name in out_df["family"].unique():
        family_mask = out_df["family"] == family_name
        family_pvals = out_df.loc[family_mask, "p_value"].tolist()
        holm_rejects = holm_bonferroni_reject(family_pvals, alpha=0.05)
        out_df.loc[family_mask, "rejects_at_holm_bonferroni_05"] = holm_rejects

    # Console log all rows
    for _, row in out_df.iterrows():
        sign_marker_bon = "*" if row["rejects_at_bonferroni_05"] else " "
        sign_marker_holm = "H" if row["rejects_at_holm_bonferroni_05"] else " "
        logger.info(
            "[%s] %s vs %s on %-18s  Δ=%+.4f  CI=[%+.4f,%+.4f]  p_boot=%.4g  p_sign=%.4g  "
            "%s%s %d/%d seeds A<B",
            row["family"], row["pivot"], row["comparator"], row["metric"],
            row["delta"], row["ci_low"], row["ci_high"],
            row["p_value"], row["p_sign"],
            sign_marker_bon, sign_marker_holm, int(row["seed_a_lt_b"]), int(row["n_common_seeds"]),
        )

    out_csv = results_dir / "significance_tests.csv"
    out_df.to_csv(out_csv, index=False, float_format="%.6g")
    logger.info("Wrote %s (total rows=%d)", out_csv, len(out_df))

    # Markdown summary
    summary_md_path = results_dir / "significance_summary.md"
    _write_summary_md(out_df, args.pivot, n_tests, alpha_corrected, summary_md_path)
    logger.info("Wrote %s", summary_md_path)
    return 0


def _write_summary_md(
    df: pd.DataFrame, pivot: str, n_tests: int, alpha_corr: float, out: Path,
) -> None:
    pivot_display = METHOD_DISPLAY.get(pivot, pivot)
    lines = [
        f"# Significance suite — pivot: {pivot_display}\n",
        f"\nPaired bootstrap (Dror et al. ACL 2018), n_resample=10,000. ",
        f"Bonferroni-corrected α = 0.05/{n_tests} = {alpha_corr:.4f} for the main family. ",
        "Tests organised in two pre-specified hypothesis families: ",
        "**main** (pivot vs comparators on all 6 metrics), ",
        "**within_arch_transfer** (large-arch CCI vs large-arch Model C on the 3 CFR metrics only). ",
        "Each family is Bonferroni-corrected separately AND Holm-corrected per family. ",
        "Reported per row: bootstrap p-value (`p_boot`), exact sign-test p-value (`p_sign`, "
        "minimum achievable under H0 on direction agreement), `n_common` (paired seeds used; "
        "5 vs 6 when one comparator dropped a seed). ",
        "Reject marker: **B** = Bonferroni, **H** = Holm-Bonferroni. ",
        "Δ = pivot − comparator; negative Δ on CFR/FPED/FNED/ECE means pivot is better.\n\n",
    ]

    metric_order = [
        "cfr_hard", "cfr_soft_l1", "cfr_soft_js",
        "fped", "fned", "ece_worst_group",
    ]

    families = list(dict.fromkeys(df["family"].tolist()))
    for family in families:
        family_df = df[df["family"] == family]
        family_pivot = family_df["pivot"].iloc[0]
        family_pivot_disp = METHOD_DISPLAY.get(family_pivot, family_pivot)
        n_family = len(family_df)
        n_reject_bon = int(family_df["rejects_at_bonferroni_05"].sum())
        n_reject_holm = int(family_df["rejects_at_holm_bonferroni_05"].sum())
        lines.append(
            f"\n## Family: `{family}` (pivot: {family_pivot_disp})\n\n"
            f"{n_family} tests; Bonferroni rejects: {n_reject_bon}; Holm-Bonferroni rejects: {n_reject_holm}.\n\n"
        )
        lines.append(
            "| Comparator | Metric | A (pivot) | B (comp) | Δ (A−B) | 95% CI | "
            "p_boot | p_sign | Bonf | Holm | n A<B / n |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        comparators = list(dict.fromkeys(family_df["comparator"].tolist()))
        for comp in comparators:
            sub = family_df[family_df["comparator"] == comp]
            comp_display = METHOD_DISPLAY.get(comp, comp)
            for metric in metric_order:
                row_match = sub[sub["metric"] == metric]
                if row_match.empty:
                    continue
                r = row_match.iloc[0]
                bon_mark = "**B**" if r["rejects_at_bonferroni_05"] else " "
                holm_mark = "**H**" if r["rejects_at_holm_bonferroni_05"] else " "
                line = (
                    f"| {comp_display} | {metric} | "
                    f"{r['metric_a']:.4f} | {r['metric_b']:.4f} | "
                    f"{r['delta']:+.4f} | "
                    f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                    f"{r['p_value']:.4g} | "
                    f"{r['p_sign']:.4g} | "
                    f"{bon_mark} | {holm_mark} | "
                    f"{int(r['seed_a_lt_b'])}/{int(r.get('n_common_seeds', 6))} |"
                )
                lines.append(line + "\n")
            lines.append("|  |  |  |  |  |  |  |  |  |  |  |\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    raise SystemExit(main())
