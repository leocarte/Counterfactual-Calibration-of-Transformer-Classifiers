#!/usr/bin/env python3
"""Does the soft-CFR inflation generalize beyond HKRR multicalibration?

The paper's headline tension: post-hoc multicalibration improves subgroup
calibration but INFLATES the soft counterfactual flip rate by 96% — because a
tweet and its identity-swapped counterfactual fall in different (group, bin)
cells, so the per-cell correction pulls their probabilities apart even when
their hard labels agree.

To show this is a property of *per-cell group-conditional post-hoc correction*
and not an artefact of the one HKRR patcher, this script applies a SECOND
group-conditional calibrator (per-group temperature scaling, or multicalibration
for replication) to a trained checkpoint and measures CFR_soft (and CFR_hard,
ECE) BEFORE and AFTER calibration over the counterfactual-swap manifest.

No training — runs on the already-trained Model C seeds. The mechanism predicts
the inflation: the original tweet and its swap map to different identity groups,
so a group-conditional calibrator applies a different correction to each side.

Usage (one Model C seed):

    python scripts/eval_calibrator_cfr.py \
        --checkpoint /scratch/$USER/antisemitism-results/checkpoints/model_c_deberta_focal_mask_seed42/best_model.pt \
        --config configs/deberta_focal_mask.yaml \
        --calibrator per_group_temp \
        --output-dir /scratch/$USER/antisemitism-results/calibrator_cfr/per_group_temp_seed42

Writes metrics.json with CFR_soft / CFR_hard / ECE pre+post and the inflation %.
Aggregate the per-seed metrics.json across seeds for the paper number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/eval_calibrator_cfr.py ...` without PYTHONPATH=. ;
# analysis/direction1 is added for _utils (shared scratch-root / manifest path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(1, str(_REPO_ROOT / "analysis" / "direction1"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.calibration import (
    MulticalibrationPatcher,
    PerGroupTemperatureScaling,
)
from src.data.dataset import AntisemitismDataset
from src.data.identity_inventory import (
    PRIMARY_KEYWORD_GROUPS,
    group_label_to_index,
    primary_keyword_group,
)
from src.data.splits import load_splits
from src.evaluation.metrics import expected_calibration_error
from src.models.transformer_classifier import TransformerClassifier
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def collect_logits(model, loader, device):
    """Stack model logits over a loader as a numpy array of shape (N, 2)."""
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        o = model(batch["input_ids"], batch["attention_mask"])
        out.append(o["logits"].detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def groups_of(texts):
    """Map each text to its primary identity-keyword group integer index."""
    return np.asarray(
        [group_label_to_index(primary_keyword_group(t)) for t in texts],
        dtype=np.int64,
    )


def build_calibrator(name, n_bins):
    num_groups = len(PRIMARY_KEYWORD_GROUPS)
    if name == "per_group_temp":
        return PerGroupTemperatureScaling(num_groups=num_groups)
    if name == "multicalibration":
        return MulticalibrationPatcher(num_groups=num_groups, n_bins=n_bins)
    raise ValueError(
        f"Unknown calibrator {name!r}; expected per_group_temp or multicalibration"
    )


def logits_for(texts, tokenizer, cfg, model, device, batch_size):
    """Forward-pass a list of texts (dummy labels) and return logits (N, 2)."""
    ds = AntisemitismDataset(
        texts=list(texts), labels=[0] * len(texts),
        tokenizer=tokenizer, max_length=cfg.data.max_length,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return collect_logits(model, loader, device)


def pos_prob(logits):
    """Positive-class probability from 2-class logits."""
    return torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Second group-conditional calibrator vs counterfactual "
                    "invariance: CFR_soft before/after calibration."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Trained .pt checkpoint (e.g. a Model C seed).")
    parser.add_argument("--config", required=True,
                        help="Training config (tokenizer + paths + model).")
    parser.add_argument("--calibrator", default="per_group_temp",
                        choices=["per_group_temp", "multicalibration"],
                        help="per_group_temp = the SECOND calibrator (the point); "
                             "multicalibration replicates the paper's HKRR result.")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="n_bins for multicalibration (ignored for per_group_temp).")
    parser.add_argument("--manifest-path", default=None,
                        help="Swap manifest parquet. Default: shared scratch manifest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- manifest (the counterfactual swap pairs) ----
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        from _utils import cfr_cache_root  # analysis/direction1 on path
        manifest_path = cfr_cache_root() / "_shared" / "test_swaps_manifest.parquet"
    if not manifest_path.exists():
        logger.error("Swap manifest not found: %s. Build it with "
                     "analysis/direction1/01_build_swap_manifest.py.", manifest_path)
        return 2
    pairs = pd.read_parquet(manifest_path)
    orig_texts = pairs["original_text"].astype(str).tolist()
    swap_texts = pairs["swap_text"].astype(str).tolist()
    logger.info("Manifest: %s (%d pairs)", manifest_path, len(pairs))

    # ---- model ----
    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)
    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type=cfg.training.loss_type,
        class_weights=list(cfg.training.get("class_weights", [1.0, 1.0])),
        focal_gamma=cfg.training.get("focal_gamma", 2.0),
        focal_alpha=cfg.training.get("focal_alpha", 0.75),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # CCI v2 checkpoints are wrapped {"model": ..., "temperature_head": ...}.
    if isinstance(state, dict) and "model" in state and "temperature_head" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    logger.info("Loaded checkpoint %s on %s", args.checkpoint, device)

    # ---- dev (fit calibrator) + test (ECE sanity) ----
    splits = load_splits(cfg.paths.processed_dir)
    dev_texts = splits["dev"][cfg.data.text_column].tolist()
    dev_y = np.asarray(splits["dev"][cfg.data.label_column].tolist(), dtype=np.int64)
    test_texts = splits["test"][cfg.data.text_column].tolist()
    test_y = np.asarray(splits["test"][cfg.data.label_column].tolist(), dtype=np.int64)

    logger.info("Forward pass: dev / test / manifest(orig) / manifest(swap)...")
    dev_logits = logits_for(dev_texts, tokenizer, cfg, model, device, args.batch_size)
    test_logits = logits_for(test_texts, tokenizer, cfg, model, device, args.batch_size)
    orig_logits = logits_for(orig_texts, tokenizer, cfg, model, device, args.batch_size)
    swap_logits = logits_for(swap_texts, tokenizer, cfg, model, device, args.batch_size)

    dev_groups = groups_of(dev_texts)
    test_groups = groups_of(test_texts)
    orig_groups = groups_of(orig_texts)
    swap_groups = groups_of(swap_texts)

    # ---- fit the group-conditional calibrator on dev ----
    cal = build_calibrator(args.calibrator, args.n_bins)
    logger.info("Fitting %s on %d dev examples...", args.calibrator, len(dev_y))
    cal.fit(dev_logits, dev_groups, dev_y)

    # ---- pairwise probabilities: pre and post calibration ----
    p_orig_pre, p_swap_pre = pos_prob(orig_logits), pos_prob(swap_logits)
    p_orig_post = cal.transform_probs(orig_logits, orig_groups)
    p_swap_post = cal.transform_probs(swap_logits, swap_groups)

    def cfr_soft(a, b):
        return float(np.mean(np.abs(a - b)))

    def cfr_hard(a, b):
        return float(np.mean((a >= 0.5) != (b >= 0.5)))

    cfr_soft_pre, cfr_soft_post = cfr_soft(p_orig_pre, p_swap_pre), cfr_soft(p_orig_post, p_swap_post)
    cfr_hard_pre, cfr_hard_post = cfr_hard(p_orig_pre, p_swap_pre), cfr_hard(p_orig_post, p_swap_post)
    inflation = (cfr_soft_post - cfr_soft_pre) / max(1e-12, cfr_soft_pre)

    # ---- ECE on test (sanity: calibration should improve / not hurt) ----
    test_pre = pos_prob(test_logits)
    test_post = cal.transform_probs(test_logits, test_groups)
    ece_pre = expected_calibration_error(list(test_y), list(test_pre))
    ece_post = expected_calibration_error(list(test_y), list(test_post))

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "calibrator": args.calibrator,
        "n_pairs": int(len(pairs)),
        "cfr_soft_l1_pre": cfr_soft_pre,
        "cfr_soft_l1_post": cfr_soft_post,
        "cfr_soft_inflation_frac": inflation,
        "cfr_hard_pre": cfr_hard_pre,
        "cfr_hard_post": cfr_hard_post,
        "ece_test_pre": ece_pre,
        "ece_test_post": ece_post,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 64)
    logger.info(
        "[%s] CFR_soft %.4f -> %.4f (%+.1f%%) | CFR_hard %.4f -> %.4f | "
        "ECE(test) %.4f -> %.4f",
        args.calibrator, cfr_soft_pre, cfr_soft_post, 100 * inflation,
        cfr_hard_pre, cfr_hard_post, ece_pre, ece_post,
    )
    logger.info("Wrote %s", out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
