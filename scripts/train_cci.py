#!/usr/bin/env python3
"""Training script for Counterfactual Calibration Invariance (CCI v2).

Mirrors scripts/train_clp.py but swaps the CLP loss for the CCI v2 loss with a
:class:`TemperatureHead` (per-group T, learnable or fixed) and JS or L1 pair
divergence. Records ``T_g`` per epoch and dumps the trajectory for the
post-hoc base-rate-shortcut analysis.

Usage:
    python scripts/train_cci.py --config configs/cci_v2_learned_js.yaml --seed 42

Required config keys (under ``training``):
    lambda_cci             : float, weight on CCI penalty
    cci_temperature_learnable : bool, learnable per-group T or fixed buffer
    cci_divergence         : "l1" or "js"
    cci_init_T             : float (default 1.0)
    cci_T_min, cci_T_max   : float, clip range (default 0.5, 5.0)

The same cf_swaps_per_token / cf_pairs_cap_per_example / cf_symmetry_filter
config knobs as train_clp.py are honoured (the swap engine is shared).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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
from src.data.identity_inventory import (
    PRIMARY_KEYWORD_GROUPS,
    group_label_to_index,
    primary_keyword_group,
)
from src.models.transformer_classifier import TransformerClassifier
from src.models.temperature_head import TemperatureHead
from src.training.cci_loss import CCIv2Loss
from src.training.cci_trainer import CCITrainer
from src.training.cci_diagnostics import (
    analyse_temperature_shortcut,
    compute_per_group_base_rates,
)
from src.evaluation.metrics import compute_metrics, per_keyword_metrics

logger = get_logger(__name__)


def get_num_workers() -> int:
    if sys.platform == "win32":
        return 0
    return 0  # Same rationale as train_clp.py — keep simple.


class _NoOpSymmetryClassifier:
    def decide(self, x, x_swap, swap_pair):
        from src.data.symmetry_classifier import SymmetryDecision
        return SymmetryDecision(True, "no_filter_ablation")
    def is_symmetric(self, x, x_swap, swap_pair):
        return True


def build_swap_engine(cfg, seed: int) -> CounterfactualSwapEngine:
    swaps_per_token = int(cfg.training.get("cf_swaps_per_token", 3))
    pairs_cap = int(cfg.training.get("cf_pairs_cap_per_example", 5))
    filter_kind = cfg.training.get("cf_symmetry_filter", "heuristic")
    if filter_kind == "heuristic":
        classifier = HeuristicSymmetryClassifier()
    elif filter_kind in {"none", "no_filter", "off"}:
        classifier = _NoOpSymmetryClassifier()
    else:
        raise ValueError(f"Unknown cf_symmetry_filter={filter_kind!r}")
    return CounterfactualSwapEngine(
        classifier=classifier,
        swaps_per_token=swaps_per_token,
        pairs_cap_per_example=pairs_cap,
        rng=random.Random(seed),
    )


def main():
    parser = argparse.ArgumentParser(description="Train CCI v2 model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.experiment_name = f"{cfg.experiment_name}_seed{args.seed}"
    seed_everything(cfg.seed)

    # Required CCI knobs
    for key in ("lambda_cci", "cci_temperature_learnable", "cci_divergence"):
        if key not in cfg.training:
            logger.error("Config missing training.%s", key)
            sys.exit(2)

    lambda_cci = float(cfg.training.lambda_cci)
    cci_learnable = bool(cfg.training.cci_temperature_learnable)
    cci_divergence = str(cfg.training.cci_divergence).lower()
    cci_init_T = float(cfg.training.get("cci_init_T", 1.0))
    cci_T_min = float(cfg.training.get("cci_T_min", 0.5))
    cci_T_max = float(cfg.training.get("cci_T_max", 5.0))

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(
        "CCI v2 config: lambda=%.3f learnable_T=%s divergence=%s "
        "T_init=%.2f T_min=%.2f T_max=%.2f",
        lambda_cci, cci_learnable, cci_divergence, cci_init_T, cci_T_min, cci_T_max,
    )

    if not args.no_wandb:
        try:
            wandb.init(project=cfg.wandb.project, name=cfg.experiment_name, config=dict(cfg))
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Continuing without logging.")
            args.no_wandb = True

    # Splits
    processed_dir = Path(cfg.paths.processed_dir)
    if (processed_dir / "train.csv").exists():
        splits = load_splits(cfg.paths.processed_dir)
    else:
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

    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)

    # Swap engine + CF train dataset
    swap_engine = build_swap_engine(cfg, seed=cfg.seed)
    train_texts = splits["train"][cfg.data.text_column].tolist()
    train_labels = splits["train"][cfg.data.label_column].tolist()

    # Per-group base rates on train (for the post-hoc shortcut analysis)
    train_group_ids = [
        group_label_to_index(primary_keyword_group(t)) for t in train_texts
    ]
    base_rates = compute_per_group_base_rates(
        labels=train_labels,
        group_ids=train_group_ids,
        num_groups=len(PRIMARY_KEYWORD_GROUPS),
    )
    logger.info("Per-group base rates on train: %s",
                {g: f"{r:.3f}" for g, r in zip(PRIMARY_KEYWORD_GROUPS, base_rates)})

    # Diagnostic: pair yield
    yield_stats = compute_swap_yield(train_texts, swap_engine)
    logger.info(
        "Swap yield: %d/%d texts have identity tokens; %d yield ≥1 symmetric pair "
        "(mean %.2f pairs/identity-text)",
        yield_stats["texts_with_identity"], yield_stats["total_texts"],
        yield_stats["texts_with_pairs"], yield_stats["mean_pairs_per_identity_text"],
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
        return_source_group=True,
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
        collate_fn=cf_collate_fn,
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
    logger.info(f"Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")

    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type=cfg.training.loss_type,  # "focal" for CCI v2
        class_weights=list(cfg.training.get("class_weights", [1.0, 1.0])),
        focal_gamma=cfg.training.get("focal_gamma", 2.0),
        focal_alpha=cfg.training.get("focal_alpha", 0.75),
    )

    # CCI v2 components
    temperature_head = TemperatureHead(
        num_groups=len(PRIMARY_KEYWORD_GROUPS),
        init_T=cci_init_T,
        T_min=cci_T_min,
        T_max=cci_T_max,
        learnable=cci_learnable,
    )
    cci_loss_fn = CCIv2Loss(
        temperature_head=temperature_head,
        divergence=cci_divergence,
        positive_class_index=int(cfg.training.get("cci_positive_class_index", 1)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    training_config = dict(cfg.training)

    trainer = CCITrainer(
        model=model,
        temperature_head=temperature_head,
        cci_loss_fn=cci_loss_fn,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=training_config,
        device=device,
        lambda_cci=lambda_cci,
    )

    save_dir = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name
    history = trainer.train(str(save_dir))

    # Test set
    logger.info("=" * 50)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 50)
    test_metrics = trainer.evaluate(test_loader)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Per-keyword
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

    # Persist outputs
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

    # Post-hoc base-rate-shortcut analysis (the kill criterion)
    shortcut_verdict = analyse_temperature_shortcut(
        trajectory=trainer.temperature_trajectory,
        base_rates=base_rates,
    )
    logger.info("Temperature-shortcut verdict: %s", shortcut_verdict["reason"])
    if shortcut_verdict["shortcut_detected"]:
        logger.warning(
            "BASE-RATE SHORTCUT DETECTED — CCI v2 may be acting as a label-shift "
            "correction. See the design doc Kill Criterion."
        )

    results_dir = Path(cfg.paths.metrics_dir) / cfg.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    with open(results_dir / "temperature_shortcut_verdict.json", "w") as f:
        json.dump(shortcut_verdict, f, indent=2, default=str)
    save_config(cfg, str(results_dir / "config.yaml"))

    if wandb.run is not None:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.log({
            "shortcut_detected": int(shortcut_verdict["shortcut_detected"]),
            "var_T_final": shortcut_verdict["var_T_final"],
            "corr_T_inv_baserate": shortcut_verdict["corr_T_inv_baserate"],
        })
        wandb.finish()

    logger.info(f"Results saved to {results_dir}")
    logger.info("CCI v2 training complete.")


if __name__ == "__main__":
    main()
