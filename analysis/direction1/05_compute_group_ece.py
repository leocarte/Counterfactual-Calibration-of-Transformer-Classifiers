#!/usr/bin/env python3
"""Per-group Expected Calibration Error per checkpoint.

Standard equal-width ECE (Guo et al. ICML 2017, n_bins=15) computed on the
positive-class probability ``p = σ(logit_1 − logit_0)`` of each test
example, partitioned by the dataset's keyword group (Jews / Israel /
Kikes / ZioNazi). Produces:

  * ``per_checkpoint_group_ece.parquet`` — per-(method, seed) rows
  * ``group_ece_summary.csv``            — mean ± std per method
  * ``main_table_group_ece.md``          — paper-ready markdown

CPU-only. Reads the same caches as ``04_compute_fped_fned.py``.

Usage:
    python3 analysis/direction1/05_compute_group_ece.py
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
from src.evaluation.metrics import expected_calibration_error  # noqa: E402
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


def _per_group_ece(
    p_pos: np.ndarray,
    y_true: np.ndarray,
    keywords: np.ndarray,
    n_bins: int = 15,
    min_per_group: int = 5,
) -> dict:
    """Compute global + per-group ECE on the 4 dataset keyword groups."""
    out = {
        "ece_global": expected_calibration_error(
            y_true.tolist(), p_pos.tolist(), n_bins=n_bins,
        ),
    }
    canonical = np.asarray([normalise_keyword(k) for k in keywords])
    per_group_eces: list[float] = []
    for g in DATASET_KEYWORD_GROUPS:
        gn = normalise_keyword(g)
        mask = canonical == gn
        n_g = int(mask.sum())
        out[f"n_{gn}"] = n_g
        if n_g < min_per_group:
            out[f"ece_{gn}"] = float("nan")
            continue
        ece_g = expected_calibration_error(
            y_true[mask].tolist(), p_pos[mask].tolist(), n_bins=n_bins,
        )
        out[f"ece_{gn}"] = ece_g
        per_group_eces.append(ece_g)

    if per_group_eces:
        out["ece_worst_group"] = float(max(per_group_eces))
        out["ece_gap"] = out["ece_worst_group"] - out["ece_global"]
    else:
        out["ece_worst_group"] = float("nan")
        out["ece_gap"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Per-method drivers
# ---------------------------------------------------------------------------

def _row_for_raw_checkpoint(experiment: str, seed: int) -> dict | None:
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
    y_true = df["true_label"].to_numpy(np.int64)
    keywords = df["keyword"].astype(str).to_numpy()
    metrics = _per_group_ece(p_pos, y_true, keywords)
    return {"method": experiment, "method_kind": "trained", "seed": seed, **metrics}


def _row_for_model_c_plus_ts(seed: int) -> dict | None:
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
    y_true = test_df["true_label"].to_numpy(np.int64)
    keywords = test_df["keyword"].astype(str).to_numpy()
    metrics = _per_group_ece(p_pos_ts, y_true, keywords)
    return {
        "method": "model_c_plus_ts",
        "method_kind": "posthoc",
        "seed": seed,
        "T_fitted": float(T),
        **metrics,
    }


def _row_for_model_c_plus_multicalibration(seed: int, n_bins_patch: int = 10) -> dict | None:
    cache_dir = cfr_cache_root() / f"{BASELINE_METHODS[0]}_seed{seed}"
    test_path = cache_dir / "test_originals.parquet"
    dev_path = cache_dir / "dev_originals.parquet"
    if not (test_path.exists() and dev_path.exists()):
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
        num_groups=len(PRIMARY_KEYWORD_GROUPS), n_bins=n_bins_patch,
    ).fit(dev_logits, dev_group_ids, dev_labels)
    p_pos_mc = patcher.transform_probs(test_logits, test_group_ids)
    y_true = test_df["true_label"].to_numpy(np.int64)
    keywords = test_df["keyword"].astype(str).to_numpy()
    metrics = _per_group_ece(p_pos_mc, y_true, keywords)
    return {
        "method": "model_c_plus_multicalibration",
        "method_kind": "posthoc",
        "seed": seed,
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
                    "[raw] %s seed=%d  ECE=%.4f  worst=%.4f  gap=%.4f",
                    exp, seed, row["ece_global"], row["ece_worst_group"], row["ece_gap"],
                )

    if skip_posthoc:
        return rows

    for seed in SEEDS:
        row_ts = _row_for_model_c_plus_ts(seed)
        if row_ts is not None:
            rows.append(row_ts)
            logger.info(
                "[posthoc] model_c_plus_ts seed=%d  ECE=%.4f  worst=%.4f",
                seed, row_ts["ece_global"], row_ts["ece_worst_group"],
            )
        row_mc = _row_for_model_c_plus_multicalibration(seed)
        if row_mc is not None:
            rows.append(row_mc)
            logger.info(
                "[posthoc] model_c_plus_multicalibration seed=%d  ECE=%.4f  worst=%.4f",
                seed, row_mc["ece_global"], row_mc["ece_worst_group"],
            )

    return rows


def _agg_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ece_global", "ece_worst_group", "ece_gap",
        *(f"ece_{normalise_keyword(g)}" for g in DATASET_KEYWORD_GROUPS),
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
        rows.append(row)
    return pd.DataFrame(rows)


def _build_main_table_md(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Direction-1 main table — Group-conditional ECE\n",
             "Mean ± std across seeds (n=6 unless otherwise noted). ",
             "ECE computed with 15 equal-width bins (Guo ICML 2017). ",
             "Per-group ECE uses the dataset's keyword grouping; ",
             "worst-group ECE is the max across the 4 groups.\n\n"]

    lines.append("| Method | n_seeds | ECE_global | ECE_worst | ECE_gap |\n")
    lines.append("|---|---|---|---|---|\n")

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
            f"{_fmt(r, 'ece_global')} | "
            f"{_fmt(r, 'ece_worst_group')} | "
            f"{_fmt(r, 'ece_gap')} |\n"
        )

    lines.append("\n## Per-group ECE breakdown\n\n")
    cols = [normalise_keyword(g) for g in DATASET_KEYWORD_GROUPS]
    lines.append("| Method | " + " | ".join(g.title() for g in cols) + " |\n")
    lines.append("|" + "---|" * (len(cols) + 1) + "\n")
    for method in method_order:
        if method not in by_method:
            continue
        r = by_method[method]
        cells = [METHOD_DISPLAY[method]]
        for g in cols:
            cells.append(_fmt(r, f"ece_{g}"))
        lines.append("| " + " | ".join(cells) + " |\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute group-conditional ECE per checkpoint.")
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
    df.to_parquet(output_dir / "per_checkpoint_group_ece.parquet", index=False)
    logger.info("Wrote %s (%d rows)",
                output_dir / "per_checkpoint_group_ece.parquet", len(df))

    summary = _agg_table(df)
    summary.to_csv(output_dir / "group_ece_summary.csv", index=False)
    logger.info("Wrote %s", output_dir / "group_ece_summary.csv")

    _build_main_table_md(summary, output_dir / "main_table_group_ece.md")
    logger.info("Wrote %s", output_dir / "main_table_group_ece.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
