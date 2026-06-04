#!/usr/bin/env python3
"""Tier-1A threshold-optimal cross-dataset evaluation.

Cross-dataset evaluation currently scores every model at the default decision
threshold p=0.5. PR-AUC says CCI v2 has equal-or-better ranking quality
cross-dataset than Model C, but at p=0.5 CCI v2 sits at a more conservative
operating point so F1+ is lower. This script falsifies the
"threshold-shift artifact, not a representation deficit" hypothesis by
sweeping the decision threshold per (model, arch, seed, dataset) cell and
reporting F1+ at three operating points:

  F1+@0.5            : default cutoff (deployable, what users see)
  F1+@source-tuned   : threshold tuned on the GoldStandard2024 source test
                       split (predictions_test.csv), then APPLIED unchanged
                       to OOD. Deployable: only source-side info is used at
                       calibration time. This is the honest middle column.
  F1+@opt            : threshold tuned on the OOD set itself. NOT deployable
                       (test-on-test leak); sets the upper bound on what
                       threshold tuning could ever recover from F1+@0.5.

Pipeline:
  1. For each (model, arch, seed) checkpoint:
     - Load predictions_test.csv (in-distribution test predictions, written
       by scripts/evaluate.py) and sweep threshold on it to find t_source.
       Skip this cell's source-tuned column if predictions_test.csv missing.
  2. For each (model, arch, seed) x (hatexplain_jewish, toxigen_jewish):
     - If predictions_<dataset>.csv exists in the matching cross-dataset
       output dir (written by the patched src/evaluation/cross_dataset.py),
       load it.
     - Otherwise, load the checkpoint and run inference to produce probs.
  3. Sweep threshold from 0.05 to 0.95 in 0.01 steps; record argmax F1+.
  4. Apply t_source (from step 1) to compute the source-tuned column.
  5. Write a per-cell CSV and print a (model, arch)-grouped summary
     (mean / std over seeds) to stdout.

Caveats:
  - F1+@opt optimises threshold on the OOD set itself, so it is a *ceiling*
    on threshold-tuning recovery, NOT a deployable number. State this in the
    paper.
  - F1+@source-tuned uses only source-side info at calibration time and IS
    deployable. The gap (F1+@source-tuned - F1+@0.5) vs (F1+@opt - F1+@0.5)
    measures how much of the threshold-shift gap can be closed without OOD
    labels — the "in principle" vs "in practice" comparison.
  - The aggregator silently ignores cells where neither predictions.csv nor a
    checkpoint is available, and prints a "missing cells" report at the end so
    you can see what got skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow `python scripts/threshold_sweep_cross_dataset.py ...` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import AntisemitismDataset
from src.evaluation.cross_dataset import (
    load_hatexplain_jewish_subset,
    load_toxigen_jewish_subset,
)
from src.models.transformer_classifier import TransformerClassifier
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


# (model_tag, arch_tag, config_filename, checkpoint_dirname_prefix)
# `checkpoint_dirname_prefix` matches the layout used by submit_cross_dataset.sh
# and train.py: <scratch>/checkpoints/<prefix>_seed<N>/best_model.pt.
MODEL_VARIANTS = [
    ("model_c", "deberta-base",  "deberta_focal_mask.yaml",                 "model_c_deberta_focal_mask"),
    ("cci_v2",  "deberta-base",  "cci_v2_fixed_js.yaml",                    "cci_v2_fixed_js"),
    ("model_c", "deberta-large", "model_c_deberta_v3_large.yaml",           "model_c_deberta_v3_large"),
    ("cci_v2",  "deberta-large", "cci_v2_fixed_js_deberta_v3_large.yaml",   "cci_v2_fixed_js_deberta_v3_large"),
    ("model_c", "roberta-large", "model_c_roberta_large.yaml",              "model_c_roberta_large"),
    ("cci_v2",  "roberta-large", "cci_v2_fixed_js_roberta_large.yaml",      "cci_v2_fixed_js_roberta_large"),
]

DATASETS = [
    ("hatexplain", "predictions_hatexplain.csv", load_hatexplain_jewish_subset),
    ("toxigen",    "predictions_toxigen.csv",    load_toxigen_jewish_subset),
]

SEEDS = [42, 43, 44, 45, 46, 47]

# DeBERTa-large + CCI v2 was trained on 5 seeds (47 dropped per REPORT.md).
SEED_OVERRIDES = {
    "cci_v2_fixed_js_deberta_v3_large": [42, 43, 44, 45, 46],
}


@dataclass
class Cell:
    """One (model, arch, seed, dataset) cell awaiting threshold sweep."""
    model_tag: str
    arch_tag: str
    seed: int
    dataset_tag: str
    probs: np.ndarray
    labels: np.ndarray
    source: str  # "predictions_csv" or "checkpoint_inference"
    # Threshold tuned on the in-distribution source test set for this
    # (model, arch, seed) — same value for both dataset_tags of a given source
    # cell, because it depends only on source-side data. None if
    # predictions_test.csv was not found on disk.
    source_tuned_threshold: Optional[float] = None


def _find_predictions_csv(
    metrics_root: Path,
    checkpoint_prefix: str,
    seed: int,
    predictions_filename: str,
) -> Optional[Path]:
    """Locate predictions_<dataset>.csv for this (model, seed) on disk.

    Searches the three layouts in use:
      - results/metrics_from_rcp/cross_dataset_<prefix>_seed<N>/<prefix>/<predictions>.csv
        (current layout — `scripts/evaluate.py:96` does
        `output_dir = Path(args.output_dir) / cfg.experiment_name`, so the
        --output-dir passed by submit_cross_dataset.sh gets an extra
        <experiment_name> level appended; experiment_name happens to equal
        the checkpoint prefix for every config we use)
      - results/metrics_from_rcp/cross_dataset_<prefix>_seed<N>/<predictions>.csv
        (the layout I originally expected — kept as a fallback in case a
        future evaluate.py refactor drops the experiment_name suffix)
      - results/metrics_from_rcp/cross_dataset_seed<N>/<prefix>/<predictions>.csv
        (used by the original Model C base run from commit 7a7e0fc)
    """
    candidates = [
        metrics_root / f"cross_dataset_{checkpoint_prefix}_seed{seed}" / checkpoint_prefix / predictions_filename,
        metrics_root / f"cross_dataset_{checkpoint_prefix}_seed{seed}" / predictions_filename,
        metrics_root / f"cross_dataset_seed{seed}" / checkpoint_prefix / predictions_filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _compute_source_tuned_threshold(
    metrics_root: Path,
    checkpoint_prefix: str,
    seed: int,
    grid: np.ndarray,
) -> Optional[float]:
    """Return the F1+-optimal threshold on the in-distribution source test set
    for this (model, seed), or None if predictions_test.csv is missing.

    The CSV is written by scripts/evaluate.py:99-110 (commit 87ab5fc) and
    contains text, label, prob_positive, pred_at_0.5 for each example in
    GoldStandard2024's held-out test split. Choosing the F1+-optimal
    threshold here uses ONLY source-side information at calibration time,
    so the resulting threshold (when applied unchanged to OOD) gives an
    honestly deployable F1+@source-tuned column for the §4 transfer table.

    This is leaky w.r.t. the in-distribution test metric itself (the same
    test set was used to score CCI v2's in-dist F1+ in the paper), but that
    leak is irrelevant for the cross-dataset claim — the same test split
    is held out from training in every (model, seed) cell, so it is a fair
    source-side calibration set for OOD inference.
    """
    # Same three-layout search as _find_predictions_csv. See that function's
    # docstring for why the current layout has the extra <experiment_name>
    # subdir level.
    candidates = [
        metrics_root / f"cross_dataset_{checkpoint_prefix}_seed{seed}" / checkpoint_prefix / "predictions_test.csv",
        metrics_root / f"cross_dataset_{checkpoint_prefix}_seed{seed}" / "predictions_test.csv",
        metrics_root / f"cross_dataset_seed{seed}" / checkpoint_prefix / "predictions_test.csv",
    ]
    for c in candidates:
        if not c.exists():
            continue
        df = pd.read_csv(c)
        probs = df["prob_positive"].to_numpy(dtype=np.float64)
        labels = df["label"].to_numpy(dtype=np.int64)
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        return best_t
    return None


def _run_inference_from_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    dataset_df: pd.DataFrame,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a full forward pass over `dataset_df` and return (probs, labels).

    Mirrors scripts/evaluate.py: same TransformerClassifier construction, the
    same CCI v2 {model, temperature_head} unwrap, the same AntisemitismDataset
    wrapping. We don't want this to diverge from scripts/evaluate.py — if it
    does, the predictions.csv generated by patched cross_dataset.py and the
    on-the-fly inference here would silently disagree.
    """
    cfg = load_config(str(config_path))
    seed_everything(cfg.seed if hasattr(cfg, "seed") and cfg.seed is not None else seed)

    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)
    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type=cfg.training.loss_type,
        class_weights=cfg.training.get("class_weights", None),
    )
    # Unwrap wrapped CCI v2 ckpt: {"model", "temperature_head"} — head is
    # training-only. Same logic as scripts/evaluate.py:58-67 (commit b1eeabe).
    ckpt = torch.load(str(checkpoint_path), weights_only=True, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and "temperature_head" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            f"Architecture mismatch loading {checkpoint_path}: "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            f"Check --config matches the trained model. "
            f"First 5 missing: {missing[:5]}."
        )
    model = model.to(device)
    model.eval()

    dataset = AntisemitismDataset(
        texts=dataset_df["text"].tolist(),
        labels=dataset_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    num_workers = 0 if sys.platform == "win32" else 4
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(outputs["logits"], dim=-1)
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    # Free GPU memory before the next checkpoint loads.
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return np.asarray(all_probs, dtype=np.float64), np.asarray(all_labels, dtype=np.int64)


def _sweep_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float, float]:
    """Return (F1+ at 0.5, F1+ at optimal threshold, optimal threshold).

    F1+ is computed with sklearn's pos_label=1, zero_division=0 — matches
    src/evaluation/metrics.py:compute_metrics.
    """
    f1_at_05 = float(f1_score(labels, (probs >= 0.5).astype(int), pos_label=1, zero_division=0))
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return f1_at_05, best_f1, best_t


def collect_cells(
    metrics_root: Path,
    checkpoints_root: Optional[Path],
    configs_root: Path,
    device: str,
    cache_dataframes: dict,
    grid: np.ndarray,
) -> tuple[list[Cell], list[tuple[str, str, int, str, str]]]:
    """Walk MODEL_VARIANTS x DATASETS x SEEDS and gather (probs, labels)
    from CSV when present, else from checkpoint inference. Return cells +
    a list of cells that were skipped because nothing was available.

    Also attaches a per-(model, arch, seed) source_tuned_threshold to each
    cell, computed once per source cell from predictions_test.csv via
    `_compute_source_tuned_threshold`. None if that file is missing.
    """
    cells: list[Cell] = []
    missing: list[tuple[str, str, int, str, str]] = []
    # Memoise source-tuned thresholds per (ckpt_prefix, seed) — same value
    # gets applied to both OOD datasets for a given source cell.
    source_tuned_cache: dict[tuple[str, int], Optional[float]] = {}

    for model_tag, arch_tag, config_name, ckpt_prefix in MODEL_VARIANTS:
        seeds = SEED_OVERRIDES.get(ckpt_prefix, SEEDS)
        config_path = configs_root / config_name
        for seed in seeds:
            key = (ckpt_prefix, seed)
            if key not in source_tuned_cache:
                source_tuned_cache[key] = _compute_source_tuned_threshold(
                    metrics_root, ckpt_prefix, seed, grid
                )
            t_src = source_tuned_cache[key]

            for ds_tag, pred_fname, _loader_fn in DATASETS:
                csv_path = _find_predictions_csv(metrics_root, ckpt_prefix, seed, pred_fname)
                if csv_path is not None:
                    df = pd.read_csv(csv_path)
                    cells.append(Cell(
                        model_tag=model_tag, arch_tag=arch_tag, seed=seed,
                        dataset_tag=ds_tag,
                        probs=df["prob_positive"].to_numpy(dtype=np.float64),
                        labels=df["label"].to_numpy(dtype=np.int64),
                        source=f"predictions_csv:{csv_path}",
                        source_tuned_threshold=t_src,
                    ))
                    continue

                if checkpoints_root is None:
                    missing.append((model_tag, arch_tag, seed, ds_tag,
                                    "no predictions.csv and --checkpoints-root not given"))
                    continue

                ckpt_path = checkpoints_root / f"{ckpt_prefix}_seed{seed}" / "best_model.pt"
                if not ckpt_path.exists():
                    missing.append((model_tag, arch_tag, seed, ds_tag,
                                    f"checkpoint not found: {ckpt_path}"))
                    continue
                if not config_path.exists():
                    missing.append((model_tag, arch_tag, seed, ds_tag,
                                    f"config not found: {config_path}"))
                    continue

                # Lazy-load the external dataset df once per dataset.
                if ds_tag not in cache_dataframes:
                    cache_dataframes[ds_tag] = _loader_fn("data/external")
                dataset_df = cache_dataframes[ds_tag]
                if dataset_df.empty:
                    missing.append((model_tag, arch_tag, seed, ds_tag,
                                    "external dataset CSV missing / empty"))
                    continue

                logger.info(
                    f"Running inference: {ckpt_prefix} seed={seed} dataset={ds_tag}"
                )
                probs, labels = _run_inference_from_checkpoint(
                    config_path=config_path,
                    checkpoint_path=ckpt_path,
                    dataset_df=dataset_df,
                    device=device,
                    seed=seed,
                )
                cells.append(Cell(
                    model_tag=model_tag, arch_tag=arch_tag, seed=seed,
                    dataset_tag=ds_tag,
                    probs=probs, labels=labels,
                    source=f"checkpoint_inference:{ckpt_path}",
                    source_tuned_threshold=t_src,
                ))

    return cells, missing


def write_per_cell_csv(cells: list[Cell], grid: np.ndarray, out_path: Path) -> pd.DataFrame:
    """Sweep threshold per cell and write the long-format CSV.

    Adds F1+@source-tuned alongside F1+@0.5 and F1+@opt. The source-tuned
    threshold lives on `cell.source_tuned_threshold` (None if
    predictions_test.csv was missing for the underlying source cell).
    """
    rows = []
    for c in cells:
        f1_05, f1_opt, t_opt = _sweep_threshold(c.labels, c.probs, grid)
        # Apply source-tuned threshold UNCHANGED to OOD. None when
        # predictions_test.csv is missing for this (model, seed) — leave the
        # column blank so downstream tooling can detect & ignore.
        if c.source_tuned_threshold is None:
            f1_src = float("nan")
        else:
            f1_src = float(f1_score(
                c.labels,
                (c.probs >= c.source_tuned_threshold).astype(int),
                pos_label=1, zero_division=0,
            ))
        rows.append({
            "model": c.model_tag,
            "arch": c.arch_tag,
            "seed": c.seed,
            "dataset": c.dataset_tag,
            "n": int(len(c.labels)),
            "n_positive": int(c.labels.sum()),
            "F1+_at_0.5": f1_05,
            "F1+_at_source_tuned": f1_src,
            "F1+_at_optimal": f1_opt,
            "source_tuned_threshold": (
                float("nan") if c.source_tuned_threshold is None
                else c.source_tuned_threshold
            ),
            "optimal_threshold": t_opt,
            "delta_F1+_opt_minus_05": f1_opt - f1_05,
            "delta_F1+_src_minus_05": (
                float("nan") if c.source_tuned_threshold is None else f1_src - f1_05
            ),
            "recovery_pct": (
                # 100 * (src_gain / opt_gain): how much of the threshold-tuning
                # ceiling does source-tuning recover? NaN if opt-gain ≤ 0 or
                # source-tuned column missing.
                float("nan") if (
                    c.source_tuned_threshold is None or (f1_opt - f1_05) <= 0
                ) else 100.0 * (f1_src - f1_05) / (f1_opt - f1_05)
            ),
            "source": c.source,
        })
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "arch", "model", "seed"]
    ).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info(f"Per-cell threshold-sweep results written to {out_path}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: mean (std) of F1+ at all three operating points."""
    g = df.groupby(["dataset", "arch", "model"], sort=False)
    summary = g.agg(
        n_seeds=("seed", "count"),
        F1plus_at_05_mean=("F1+_at_0.5", "mean"),
        F1plus_at_05_std=("F1+_at_0.5", "std"),
        F1plus_src_mean=("F1+_at_source_tuned", "mean"),
        F1plus_src_std=("F1+_at_source_tuned", "std"),
        F1plus_opt_mean=("F1+_at_optimal", "mean"),
        F1plus_opt_std=("F1+_at_optimal", "std"),
        delta_opt_minus_05=("delta_F1+_opt_minus_05", "mean"),
        delta_src_minus_05=("delta_F1+_src_minus_05", "mean"),
        recovery_pct_mean=("recovery_pct", "mean"),
        source_threshold_median=("source_tuned_threshold", "median"),
        opt_threshold_median=("optimal_threshold", "median"),
    ).reset_index()
    # std with a single seed is NaN; print 0 in that case for readability.
    for c in ("F1plus_at_05_std", "F1plus_src_std", "F1plus_opt_std"):
        summary[c] = summary[c].fillna(0.0)
    return summary


def print_summary_table(summary: pd.DataFrame) -> None:
    """Pretty-print the per-arch comparison table with all three columns."""
    print()
    print("=" * 140)
    print(
        "Tier-1A: cross-dataset F1+ at three operating points "
        "(mean +/- std across seeds)"
    )
    print("=" * 140)
    print(
        f"{'dataset':<10} {'arch':<14} {'model':<8} {'n':>3} "
        f"{'F1+@0.5':>14} {'F1+@src':>14} {'F1+@opt':>14} "
        f"{'rec%':>7} {'t_src':>7} {'t_opt':>7}"
    )
    print("-" * 140)
    for _, row in summary.iterrows():
        rec_str = (
            f"{row['recovery_pct_mean']:>6.0f}%"
            if not pd.isna(row.get("recovery_pct_mean")) else "    NA"
        )
        t_src_str = (
            f"{row['source_threshold_median']:>7.2f}"
            if not pd.isna(row.get("source_threshold_median")) else "     NA"
        )
        f1_src_str = (
            f"{row['F1plus_src_mean']:.3f}+/-{row['F1plus_src_std']:.3f}"
            if not pd.isna(row.get("F1plus_src_mean")) else "      NA      "
        )
        print(
            f"{row['dataset']:<10} {row['arch']:<14} {row['model']:<8} "
            f"{int(row['n_seeds']):>3} "
            f"{row['F1plus_at_05_mean']:.3f}+/-{row['F1plus_at_05_std']:.3f}  "
            f"{f1_src_str}  "
            f"{row['F1plus_opt_mean']:.3f}+/-{row['F1plus_opt_std']:.3f}  "
            f"{rec_str} "
            f"{t_src_str} "
            f"{row['opt_threshold_median']:>7.2f}"
        )
    print("=" * 140)
    print(
        "Columns: F1+@0.5 (default cutoff), F1+@src (threshold tuned on\n"
        "GoldStandard2024's source test split, applied unchanged to OOD —\n"
        "this is deployable), F1+@opt (threshold tuned on OOD itself — UPPER\n"
        "BOUND only, not deployable). 'rec%' = 100 * (F1+@src - F1+@0.5) /\n"
        "(F1+@opt - F1+@0.5): what fraction of the threshold-tuning ceiling\n"
        "does source-side calibration recover? High rec% means the\n"
        "threshold-shift story holds 'in practice', not just 'in principle'."
    )


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description="Tier-1A: threshold-optimal cross-dataset F1+ sweep.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--metrics-root", type=str,
        default=str(here / "results" / "metrics_from_rcp"),
        help="Root that contains cross_dataset_*_seed<N>/ dirs from RCP pulls.",
    )
    ap.add_argument(
        "--checkpoints-root", type=str, default=None,
        help=(
            "If set, run inference fresh from checkpoints for any cell where\n"
            "predictions_<dataset>.csv is not yet on disk. Expected layout:\n"
            "  <checkpoints-root>/<prefix>_seed<N>/best_model.pt"
        ),
    )
    ap.add_argument(
        "--configs-root", type=str,
        default=str(here / "configs"),
    )
    ap.add_argument(
        "--out-csv", type=str,
        default=str(here / "results" / "metrics_from_rcp" / "threshold_sweep_cross_dataset.csv"),
    )
    ap.add_argument(
        "--summary-json", type=str, default=None,
        help="Optional JSON dump of the (dataset, arch, model)-summary table.",
    )
    ap.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu. Default: cuda if available, else cpu.",
    )
    ap.add_argument(
        "--threshold-min", type=float, default=0.05,
    )
    ap.add_argument(
        "--threshold-max", type=float, default=0.95,
    )
    ap.add_argument(
        "--threshold-step", type=float, default=0.01,
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )

    metrics_root = Path(args.metrics_root)
    if not metrics_root.exists():
        raise FileNotFoundError(f"--metrics-root does not exist: {metrics_root}")
    checkpoints_root = Path(args.checkpoints_root) if args.checkpoints_root else None
    configs_root = Path(args.configs_root)

    cells, missing = collect_cells(
        metrics_root=metrics_root,
        checkpoints_root=checkpoints_root,
        configs_root=configs_root,
        device=device,
        cache_dataframes={},
        grid=grid,
    )
    if not cells:
        raise RuntimeError(
            "No (model, arch, seed, dataset) cells were collected. "
            "Either re-run the cross-dataset evals so predictions_*.csv files "
            "get written, or pass --checkpoints-root pointing at a directory "
            "with the *_seed<N>/best_model.pt files."
        )

    df = write_per_cell_csv(cells, grid, Path(args.out_csv))
    summary = summarize(df)
    print_summary_table(summary)

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)
        logger.info(f"Summary JSON written to {args.summary_json}")

    if missing:
        print()
        print("Cells skipped (no predictions.csv and no checkpoint available):")
        for model_tag, arch_tag, seed, ds_tag, reason in missing:
            print(f"  {model_tag:<8} {arch_tag:<14} seed={seed} dataset={ds_tag:<10} -> {reason}")


if __name__ == "__main__":
    main()
