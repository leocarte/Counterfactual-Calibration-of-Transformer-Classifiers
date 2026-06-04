#!/usr/bin/env python3
"""Comprehensive evaluation script: test metrics, robustness, cross-dataset, error analysis."""
import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/evaluate.py ...` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.data.splits import load_splits
from src.data.dataset import AntisemitismDataset
from src.models.transformer_classifier import TransformerClassifier
from src.evaluation.metrics import compute_metrics, per_keyword_metrics
from src.evaluation.error_analysis import extract_error_cases, export_for_expert_review
from src.evaluation.robustness import (
    temporal_robustness,
    identity_term_bias_test,
)
from src.evaluation.cross_dataset import (
    load_hatexplain_jewish_subset,
    load_toxigen_jewish_subset,
    cross_dataset_evaluate,
)
from src.data.keyword_masking import keyword_sensitivity_test

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)
    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type=cfg.training.loss_type,
        class_weights=cfg.training.get("class_weights", None),
    )
    # Unwrap CCI v2 ckpt: {"model", "temperature_head"} — head is training-only.
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and "temperature_head" in ckpt:
        logger.info("Wrapped CCI v2 ckpt; loading state['model']")
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            f"Architecture mismatch: missing={len(missing)} unexpected={len(unexpected)}. "
            f"Check --config matches the trained model. First 5 missing: {missing[:5]}."
        )
    model = model.to(device)
    model.eval()

    # Load test data
    splits = load_splits(cfg.paths.processed_dir)
    test_df = splits["test"]

    test_dataset = AntisemitismDataset(
        texts=test_df["Text"].tolist(),
        labels=test_df["Biased"].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    num_workers = 0 if sys.platform == "win32" else 4
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    # Get predictions
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(outputs["logits"], dim=-1)
            all_preds.extend(outputs["logits"].argmax(-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    # Output directory
    output_dir = Path(args.output_dir or cfg.paths.metrics_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also dump in-distribution test-set per-example predictions so the
    # threshold-sweep aggregator can compute a source-tuned threshold (one
    # tuned on GoldStandard2024's test split — leaky w.r.t. in-dist eval but
    # an honest deployable middle column for the cross-dataset transfer claim:
    # how much of the OOD F1+@opt gain is recoverable from source-side info?).
    pd.DataFrame({
        "text": test_df["Text"].tolist(),
        "label": all_labels,
        "prob_positive": all_probs,
        "pred_at_0.5": all_preds,
    }).to_csv(output_dir / "predictions_test.csv", index=False)
    logger.info(f"In-dist test predictions written to {output_dir / 'predictions_test.csv'}")

    # 1. Test metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    logger.info(f"Test metrics: {json.dumps(metrics, indent=2, default=str)}")

    # 2. Per-keyword metrics
    kw_metrics = per_keyword_metrics(
        all_labels, all_preds, test_df["Keyword"].tolist()
    )
    metrics["per_keyword"] = kw_metrics

    # 3. Temporal robustness
    temporal = temporal_robustness(
        all_labels, all_preds, test_df["CreateDate"].tolist()
    )
    metrics["temporal_robustness"] = temporal
    logger.info(f"Temporal robustness: {json.dumps(temporal, indent=2)}")

    # 4. Identity term bias
    identity_bias = identity_term_bias_test(
        test_df["Text"].tolist(), all_labels, all_preds
    )
    metrics["identity_term_bias"] = identity_bias
    logger.info(f"Identity term bias: {json.dumps(identity_bias, indent=2)}")

    # 5. Keyword sensitivity test
    def predict_fn(texts):
        ds = AntisemitismDataset(texts, [0]*len(texts), tokenizer, cfg.data.max_length)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["input_ids"], batch["attention_mask"])
                preds.extend(outputs["logits"].argmax(-1).cpu().tolist())
        return preds

    kw_sensitivity = keyword_sensitivity_test(
        test_df["Text"].tolist(), all_preds, predict_fn
    )
    metrics["keyword_sensitivity"] = kw_sensitivity
    logger.info(f"Keyword sensitivity: {json.dumps(kw_sensitivity, indent=2)}")

    # 6. Error analysis
    error_cases = extract_error_cases(
        test_df["Text"].tolist(),
        all_labels, all_preds, all_probs,
        test_df["Keyword"].tolist(),
        top_n=50,
    )
    export_for_expert_review(error_cases, str(output_dir / "error_analysis.xlsx"))
    logger.info(f"Error analysis exported to {output_dir / 'error_analysis.xlsx'}")

    # 7. Cross-dataset evaluation (if external data available)
    # Also dump per-example predictions next to full_evaluation.json so the
    # Tier-1A threshold-sweep aggregator can re-score at PR-curve-optimal
    # thresholds without re-running inference.
    hatexplain_df = load_hatexplain_jewish_subset("data/external")
    if not hatexplain_df.empty:
        hx_metrics = cross_dataset_evaluate(
            model, tokenizer, hatexplain_df,
            device=device,
            predictions_out_path=output_dir / "predictions_hatexplain.csv",
        )
        metrics["cross_dataset_hatexplain"] = hx_metrics
        logger.info(f"HateXplain cross-dataset: {json.dumps(hx_metrics, indent=2, default=str)}")

    toxigen_df = load_toxigen_jewish_subset("data/external")
    if not toxigen_df.empty:
        tg_metrics = cross_dataset_evaluate(
            model, tokenizer, toxigen_df,
            device=device,
            predictions_out_path=output_dir / "predictions_toxigen.csv",
        )
        metrics["cross_dataset_toxigen"] = tg_metrics
        logger.info(f"ToxiGen cross-dataset: {json.dumps(tg_metrics, indent=2, default=str)}")

    # Save all results
    with open(output_dir / "full_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
