#!/usr/bin/env python3
"""Apply post-hoc group-conditioned calibration to a trained checkpoint.

Loads a trained transformer checkpoint, computes raw logits on dev + test,
fits one of the :class:`GroupCalibrator` implementations on dev predictions
grouped by identity-keyword, applies it to the test predictions, and writes
both pre- and post-calibration metrics so the comparison is one-look.

Usage:
    python scripts/apply_multicalibration.py \
        --checkpoint results/checkpoints/davani_lambda10_seed42/best_model.pt \
        --config configs/davani_lambda10.yaml \
        --calibrator per_group_temp \
        --output-dir results/metrics/davani_lambda10_seed42_pgT

    # Multicalibration patcher:
    python scripts/apply_multicalibration.py \
        --checkpoint <path> --config <path> \
        --calibrator multicalibration --n-bins 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/apply_multicalibration.py ...` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.calibration import (
    GroupCalibrator,
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
from src.evaluation.metrics import compute_metrics
from src.models.transformer_classifier import TransformerClassifier
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model and stack (logits, labels) as numpy arrays."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_ids"], batch["attention_mask"])
        all_logits.append(outputs["logits"].detach().cpu().float().numpy())
        all_labels.extend(batch["labels"].cpu().tolist())
    return np.concatenate(all_logits, axis=0), np.asarray(all_labels, dtype=np.int64)


def texts_to_group_ids(texts: list[str]) -> np.ndarray:
    """Map each text to its primary identity-keyword group integer index."""
    return np.asarray(
        [group_label_to_index(primary_keyword_group(t)) for t in texts],
        dtype=np.int64,
    )


def build_calibrator(name: str, n_bins: int) -> GroupCalibrator:
    num_groups = len(PRIMARY_KEYWORD_GROUPS)
    if name == "per_group_temp":
        return PerGroupTemperatureScaling(num_groups=num_groups)
    if name == "multicalibration":
        return MulticalibrationPatcher(num_groups=num_groups, n_bins=n_bins)
    raise ValueError(
        f"Unknown calibrator {name!r}; expected per_group_temp or multicalibration"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply post-hoc group calibration to a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a trained .pt checkpoint")
    parser.add_argument("--config", required=True,
                        help="Path to the training config (used for tokenizer + paths)")
    parser.add_argument("--calibrator", default="per_group_temp",
                        choices=["per_group_temp", "multicalibration"])
    parser.add_argument("--n-bins", type=int, default=10,
                        help="n_bins for the multicalibration patcher (ignored otherwise)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write {pre,post}_metrics.json + calibrator_state.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits = load_splits(cfg.paths.processed_dir)
    dev_texts = splits["dev"][cfg.data.text_column].tolist()
    test_texts = splits["test"][cfg.data.text_column].tolist()
    dev_labels = splits["dev"][cfg.data.label_column].tolist()
    test_labels = splits["test"][cfg.data.label_column].tolist()

    # Tokenizer + model
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
    # CCI v2 checkpoints (CCITrainer) are wrapped as
    #   {"model": <state_dict>, "temperature_head": <state_dict>}
    # whereas the parent Trainer saves a flat state_dict directly. Detect and
    # unwrap so the same CLI works against both formats.
    if isinstance(state, dict) and "model" in state and "temperature_head" in state:
        logger.info("Detected wrapped CCI v2 checkpoint; loading state['model'].")
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    logger.info(f"Loaded checkpoint {args.checkpoint} on {device}")

    # Datasets / loaders
    dev_ds = AntisemitismDataset(
        texts=dev_texts, labels=dev_labels,
        tokenizer=tokenizer, max_length=cfg.data.max_length,
    )
    test_ds = AntisemitismDataset(
        texts=test_texts, labels=test_labels,
        tokenizer=tokenizer, max_length=cfg.data.max_length,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Collect logits + group ids
    logger.info("Computing dev logits...")
    dev_logits, dev_y = collect_logits(model, dev_loader, device)
    dev_groups = texts_to_group_ids(dev_texts)
    logger.info("Computing test logits...")
    test_logits, test_y = collect_logits(model, test_loader, device)
    test_groups = texts_to_group_ids(test_texts)

    # Pre-calibration probabilities (standard 2-class softmax positive prob)
    pre_probs_test = torch.softmax(torch.from_numpy(test_logits), dim=-1)[:, 1].numpy()
    pre_preds_test = (pre_probs_test >= 0.5).astype(np.int64)
    pre_metrics = compute_metrics(
        list(test_y), list(pre_preds_test), list(pre_probs_test),
    )

    # Per-group ECE (pre)
    pre_per_group_ece = {}
    from src.evaluation.metrics import expected_calibration_error
    for g_name in PRIMARY_KEYWORD_GROUPS:
        g_idx = group_label_to_index(g_name)
        mask = test_groups == g_idx
        if mask.sum() < 5:
            pre_per_group_ece[g_name] = None
            continue
        pre_per_group_ece[g_name] = expected_calibration_error(
            list(test_y[mask]), list(pre_probs_test[mask])
        )

    # Fit + apply calibrator
    calibrator = build_calibrator(args.calibrator, args.n_bins)
    logger.info(f"Fitting {args.calibrator} on {len(dev_y)} dev examples...")
    calibrator.fit(dev_logits, dev_groups, dev_y)
    post_probs_test = calibrator.transform_probs(test_logits, test_groups)
    post_preds_test = (post_probs_test >= 0.5).astype(np.int64)
    post_metrics = compute_metrics(
        list(test_y), list(post_preds_test), list(post_probs_test),
    )

    # Per-group ECE (post)
    post_per_group_ece = {}
    for g_name in PRIMARY_KEYWORD_GROUPS:
        g_idx = group_label_to_index(g_name)
        mask = test_groups == g_idx
        if mask.sum() < 5:
            post_per_group_ece[g_name] = None
            continue
        post_per_group_ece[g_name] = expected_calibration_error(
            list(test_y[mask]), list(post_probs_test[mask])
        )

    # Worst-group ECE — the headline statistic for the paper's §4
    def _worst(d):
        vals = [v for v in d.values() if v is not None]
        return float(max(vals)) if vals else None
    pre_metrics["worst_group_ece"] = _worst(pre_per_group_ece)
    post_metrics["worst_group_ece"] = _worst(post_per_group_ece)
    pre_metrics["per_group_ece"] = pre_per_group_ece
    post_metrics["per_group_ece"] = post_per_group_ece

    # Persist
    with open(output_dir / "pre_metrics.json", "w") as f:
        json.dump(pre_metrics, f, indent=2, default=str)
    with open(output_dir / "post_metrics.json", "w") as f:
        json.dump(post_metrics, f, indent=2, default=str)
    with open(output_dir / "calibrator_state.json", "w") as f:
        json.dump(calibrator.export_state(), f, indent=2, default=str)

    # Calibrated test predictions for downstream analysis (Leonardo's bias audit)
    pred_df = pd.DataFrame({
        "text": test_texts,
        "true_label": test_y,
        "pre_prob": pre_probs_test,
        "post_prob": post_probs_test,
        "pre_pred": pre_preds_test,
        "post_pred": post_preds_test,
        "group": [PRIMARY_KEYWORD_GROUPS[i] for i in test_groups],
    })
    pred_df.to_csv(output_dir / "test_predictions_calibrated.csv", index=False)

    # Console summary
    logger.info("=" * 60)
    logger.info(
        f"PRE  : F1+={pre_metrics['f1_positive']:.4f}  ECE={pre_metrics['ece']:.4f}  "
        f"worst_group_ECE={pre_metrics['worst_group_ece']:.4f}"
    )
    logger.info(
        f"POST : F1+={post_metrics['f1_positive']:.4f}  ECE={post_metrics['ece']:.4f}  "
        f"worst_group_ECE={post_metrics['worst_group_ece']:.4f}"
    )
    delta_ece = post_metrics["ece"] - pre_metrics["ece"]
    logger.info(f"ΔECE = {delta_ece:+.4f}  ({delta_ece / max(1e-12, pre_metrics['ece']):.1%})")
    logger.info(f"Outputs in {output_dir}")


if __name__ == "__main__":
    main()
