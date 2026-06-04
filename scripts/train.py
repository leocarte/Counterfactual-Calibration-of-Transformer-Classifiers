#!/usr/bin/env python3
"""Main training script for transformer-based antisemitism detection.

Usage:
    python scripts/train.py --config configs/roberta.yaml
    python scripts/train.py --config configs/deberta.yaml --no-wandb
    python scripts/train.py --config configs/deberta_focal_mask.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python scripts/train.py ...` without PYTHONPATH=.
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
from src.models.transformer_classifier import TransformerClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics, per_keyword_metrics

logger = get_logger(__name__)


def get_num_workers() -> int:
    """Get appropriate num_workers based on OS and available CPUs."""
    if sys.platform == "win32":
        return 0  # Windows doesn't support fork-based multiprocessing well
    return min(4, os.cpu_count() or 1)


def main():
    parser = argparse.ArgumentParser(description="Train antisemitism detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override cfg.seed. When set, the experiment_name is also suffixed "
             "with `_seed{N}` so multi-seed runs go to separate output folders."
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Multi-seed support: override the config seed from CLI and keep outputs separated.
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.experiment_name = f"{cfg.experiment_name}_seed{args.seed}"

    seed_everything(cfg.seed)

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}")

    # Init wandb
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

    # Prepare data
    processed_dir = Path(cfg.paths.processed_dir)
    if (processed_dir / "train.csv").exists():
        logger.info("Loading existing splits...")
        splits = load_splits(cfg.paths.processed_dir)
    else:
        logger.info("Creating new splits from raw data...")
        raw_path = Path(cfg.paths.raw_data)
        if not raw_path.exists():
            logger.error(f"Raw data not found at {raw_path}. Run: python scripts/preprocess_data.py")
            sys.exit(1)

        raw_df = pd.read_csv(raw_path)

        # Preprocess
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

    # Datasets
    train_dataset = AntisemitismDataset(
        texts=splits["train"][cfg.data.text_column].tolist(),
        labels=splits["train"][cfg.data.label_column].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        keyword_masking=cfg.training.get("keyword_masking", False),
        keyword_mask_prob=cfg.training.get("keyword_mask_prob", 0.0),
        keywords=list(cfg.data.get("keywords", [])) if cfg.data.get("keywords") else None,
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
        loss_type=cfg.training.loss_type,
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

    # Training config dict
    training_config = dict(cfg.training)

    # Train
    trainer = Trainer(model, train_loader, dev_loader, training_config, device)
    save_dir = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name
    history = trainer.train(str(save_dir))

    # Final evaluation on test set
    logger.info("=" * 50)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 50)
    test_metrics = trainer.evaluate(test_loader)

    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Per-keyword evaluation
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

    # Save predictions
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
        logger.info(f"Predictions saved to {predictions_dir / 'test_predictions.csv'}")

    # Save results
    results_dir = Path(cfg.paths.metrics_dir) / cfg.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    save_config(cfg, str(results_dir / "config.yaml"))

    if wandb.run is not None:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        wandb.finish()

    logger.info(f"Results saved to {results_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
