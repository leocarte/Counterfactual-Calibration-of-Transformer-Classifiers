#!/usr/bin/env python3
"""Training script for CLP / Davani-style training (Phase 1 of training-experiments).

Mirrors :mod:`scripts.train` but swaps in:
  * :class:`CounterfactualSwapEngine` for online identity-swap generation
  * :class:`CounterfactualDataset` for the train split (dev/test stay vanilla)
  * :func:`cf_collate_fn` for variable-length cf-list collation
  * :class:`CLPTrainer` for focal + λ·CLP optimization

Usage:
    python scripts/train_clp.py --config configs/davani_lambda10.yaml
    python scripts/train_clp.py --config configs/davani_lambda10.yaml --seed 43

The config must include the new ``training.lambda_clp`` field. Optional fields:
  ``training.cf_swaps_per_token`` (default 3),
  ``training.cf_pairs_cap_per_example`` (default 5),
  ``training.cf_symmetry_filter`` ("heuristic" or "none"; default "heuristic").
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# Allow `python scripts/train_clp.py ...` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import wandb
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.data.preprocessing import preprocess_tweet
from src.data.splits import create_splits, save_splits, load_splits
from src.data.dataset import AntisemitismDataset
from src.data.cf_dataset import CounterfactualDataset, cf_collate_fn
from src.data.counterfactual_swap import CounterfactualSwapEngine, compute_swap_yield
from src.data.symmetry_classifier import HeuristicSymmetryClassifier
from src.models.transformer_classifier import TransformerClassifier
from src.training.clp_loss import CLPLoss
from src.training.clp_trainer import CLPTrainer
from src.evaluation.metrics import compute_metrics, per_keyword_metrics

logger = get_logger(__name__)


def get_num_workers() -> int:
    """Pick num_workers safely. Windows + cf_collate_fn don't play nicely with
    multiprocessing because the engine RNG would need careful per-worker seeding;
    default to 0 there. On Linux we keep 0 too for now since the collate function
    instantiates per-batch tokenizer calls — multi-worker won't help much."""
    if sys.platform == "win32":
        return 0
    return 0  # CLP path: keep simple, single-worker. Re-evaluate if profile shows bottleneck.


class _NoOpSymmetryClassifier:
    """Used by the ``no_filter`` ablation cell — accepts every swap as symmetric."""
    def decide(self, x, x_swap, swap_pair):
        from src.data.symmetry_classifier import SymmetryDecision
        return SymmetryDecision(True, "no_filter_ablation")
    def is_symmetric(self, x, x_swap, swap_pair):
        return True


def build_swap_engine(cfg, seed: int) -> CounterfactualSwapEngine:
    """Construct the swap engine from config + per-run seed."""
    swaps_per_token = int(cfg.training.get("cf_swaps_per_token", 3))
    pairs_cap = int(cfg.training.get("cf_pairs_cap_per_example", 5))
    filter_kind = cfg.training.get("cf_symmetry_filter", "heuristic")

    if filter_kind == "heuristic":
        classifier = HeuristicSymmetryClassifier()
    elif filter_kind in {"none", "no_filter", "off"}:
        classifier = _NoOpSymmetryClassifier()
        logger.warning(
            "Symmetry filter DISABLED (cf_symmetry_filter=%s). This is the "
            "no-filter ablation; expect more pairs but lower-quality CLP signal.",
            filter_kind,
        )
    else:
        raise ValueError(
            f"Unknown cf_symmetry_filter={filter_kind!r}; expected 'heuristic' or 'none'"
        )

    rng = random.Random(seed)
    return CounterfactualSwapEngine(
        classifier=classifier,
        swaps_per_token=swaps_per_token,
        pairs_cap_per_example=pairs_cap,
        rng=rng,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CLP / Davani model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cfg.seed; experiment_name suffixed with _seed{N}.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.experiment_name = f"{cfg.experiment_name}_seed{args.seed}"

    seed_everything(cfg.seed)

    # Sanity-check config has CLP-specific knobs
    if "lambda_clp" not in cfg.training:
        logger.error(
            "Config %s is missing training.lambda_clp; this script requires CLP-specific configs.",
            args.config,
        )
        sys.exit(2)
    lambda_clp = float(cfg.training.lambda_clp)

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"lambda_clp = {lambda_clp}")
    logger.info(f"cf_symmetry_filter = {cfg.training.get('cf_symmetry_filter', 'heuristic')}")

    # WandB
    if not args.no_wandb:
        try:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.experiment_name,
                config=dict(cfg),
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Continuing without logging.")
            args.no_wandb = True

    # Splits
    processed_dir = Path(cfg.paths.processed_dir)
    if (processed_dir / "train.csv").exists():
        logger.info("Loading existing splits...")
        splits = load_splits(cfg.paths.processed_dir)
    else:
        logger.info("Creating new splits from raw data...")
        raw_path = Path(cfg.paths.raw_data)
        if not raw_path.exists():
            logger.error(f"Raw data not found at {raw_path}.")
            sys.exit(1)
        raw_df = pd.read_csv(raw_path)
        raw_df[cfg.data.text_column] = raw_df[cfg.data.text_column].apply(
            lambda x: preprocess_tweet(
                x,
                mask_usernames=cfg.preprocessing.mask_usernames,
                remove_urls=cfg.preprocessing.remove_urls,
                remove_hashtag_symbol=cfg.preprocessing.remove_hashtag_symbol,
                lowercase=cfg.preprocessing.lowercase,
            )
        )
        splits = create_splits(
            raw_df,
            label_col=cfg.data.label_column,
            keyword_col=cfg.data.keyword_column,
            train_ratio=cfg.data.train_ratio,
            dev_ratio=cfg.data.dev_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.seed,
        )
        save_splits(splits, cfg.paths.processed_dir)

    # Tokenizer
    logger.info(f"Loading tokenizer for {cfg.model.name}...")
    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)

    # Swap engine + counterfactual train dataset
    swap_engine = build_swap_engine(cfg, seed=cfg.seed)
    train_texts = splits["train"][cfg.data.text_column].tolist()
    train_labels = splits["train"][cfg.data.label_column].tolist()

    # Diagnostic: how many pairs is the engine producing?
    yield_stats = compute_swap_yield(train_texts, swap_engine)
    logger.info(
        "Swap yield: %d/%d texts have identity tokens; %d yield ≥1 symmetric pair "
        "(mean %.2f pairs/identity-text)",
        yield_stats["texts_with_identity"],
        yield_stats["total_texts"],
        yield_stats["texts_with_pairs"],
        yield_stats["mean_pairs_per_identity_text"],
    )
    if not args.no_wandb and wandb.run is not None:
        wandb.log({f"swap_yield_{k}": v for k, v in yield_stats.items()
                   if isinstance(v, (int, float))})

    train_dataset = CounterfactualDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        swap_engine=swap_engine,
        max_length=cfg.data.max_length,
        keyword_masking=cfg.training.get("keyword_masking", False),
        keyword_mask_prob=cfg.training.get("keyword_mask_prob", 0.0),
        keywords=list(cfg.data.get("keywords", [])) if cfg.data.get("keywords") else None,
        return_source_group=True,  # harmless for CLP, required for CCI v2 reuse
    )
    dev_dataset = AntisemitismDataset(
        texts=splits["dev"][cfg.data.text_column].tolist(),
        labels=splits["dev"][cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    test_dataset = AntisemitismDataset(
        texts=splits["test"][cfg.data.text_column].tolist(),
        labels=splits["test"][cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )

    num_workers = get_num_workers()
    batch_size = cfg.training.batch_size

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        collate_fn=cf_collate_fn,  # critical: variable-length cf padding
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Dev: {len(dev_dataset)} samples | Test: {len(test_dataset)} samples")

    # Model
    logger.info(f"Loading model {cfg.model.name}...")
    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type=cfg.training.loss_type,  # typically "focal" for Davani
        class_weights=list(cfg.training.get("class_weights", [1.0, 1.0])),
        focal_gamma=cfg.training.get("focal_gamma", 2.0),
        focal_alpha=cfg.training.get("focal_alpha", 0.75),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.info("Using CPU (no CUDA available)")

    training_config = dict(cfg.training)

    # CLP trainer
    clp_loss_fn = CLPLoss(
        divergence=cfg.training.get("clp_divergence", "l1"),
        positive_class_index=int(cfg.training.get("clp_positive_class_index", 1)),
    )
    trainer = CLPTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=training_config,
        device=device,
        lambda_clp=lambda_clp,
        clp_loss_fn=clp_loss_fn,
    )

    save_dir = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name
    history = trainer.train(str(save_dir))

    # Test set evaluation
    logger.info("=" * 50)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 50)
    test_metrics = trainer.evaluate(test_loader)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Per-keyword (vanilla, no CF — eval-time)
    all_preds, all_labels, all_probs = [], [], []
    if cfg.evaluation.get("per_keyword", False):
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast(
                    "cuda", dtype=trainer.autocast_dtype, enabled=trainer.autocast_enabled
                ):
                    outputs = model(batch["input_ids"], batch["attention_mask"])
                probs = torch.softmax(outputs["logits"], dim=-1)
                all_preds.extend(outputs["logits"].argmax(-1).cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())

        kw_metrics = per_keyword_metrics(
            all_labels, all_preds,
            splits["test"][cfg.data.keyword_column].tolist(),
        )
        logger.info("Per-keyword metrics:")
        for kw, kw_m in kw_metrics.items():
            logger.info(f"  {kw}: F1={kw_m['macro_f1']:.4f}, n={kw_m['n']}, "
                        f"pos_rate={kw_m['base_rate']:.1%}")
        test_metrics["per_keyword"] = kw_metrics

    # Save predictions + metrics
    predictions_dir = Path(cfg.paths.predictions_dir) / cfg.experiment_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if all_preds:
        pred_df = pd.DataFrame({
            "text": splits["test"][cfg.data.text_column].tolist(),
            "true_label": all_labels,
            "predicted_label": all_preds,
            "confidence": all_probs,
            "keyword": splits["test"][cfg.data.keyword_column].tolist(),
        })
        pred_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    results_dir = Path(cfg.paths.metrics_dir) / cfg.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    save_config(cfg, str(results_dir / "config.yaml"))

    if wandb.run is not None:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    logger.info(f"Results saved to {results_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
