#!/usr/bin/env python3
"""Run LLM prompting experiments (Models D1, D2, D3)."""
import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_llm.py ...` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.data.splits import load_splits
from src.models.llm_prompter import LLMPrompter
from src.evaluation.metrics import compute_metrics, per_keyword_metrics

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run LLM prompting for antisemitism detection")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit test samples (overrides cfg.llm.limit). Use for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    # Load test data only (LLM inference is expensive)
    splits = load_splits(cfg.paths.processed_dir)
    eval_split = cfg.evaluation.get("eval_split", "test")
    test_df = splits[eval_split]

    # Apply sample limit: CLI flag overrides config value
    limit = args.limit if args.limit is not None else cfg.llm.get("limit", None)
    if limit is not None and limit < len(test_df):
        test_df = test_df.head(limit).reset_index(drop=True)
        logger.info(f"LIMIT active: using first {limit} samples (smoke-test mode)")

    texts = test_df["Text"].tolist()
    labels = test_df["Biased"].tolist()
    keywords = test_df["Keyword"].tolist()

    logger.info(f"Running LLM inference on {len(texts)} {eval_split} samples...")

    # Initialize prompter
    prompter = LLMPrompter(
        model_name=cfg.llm.model_name,
        prompt_template=cfg.llm.prompt_template,
        quantization=cfg.llm.quantization,
        max_new_tokens=cfg.llm.max_new_tokens,
        temperature=cfg.llm.temperature,
    )

    # Run inference
    results = prompter.classify_batch(texts)

    # Parse predictions
    label_map = {"antisemitic": 1, "not antisemitic": 0}
    preds = [label_map.get(r["prediction"], -1) for r in results]

    # Filter invalid responses
    valid_mask = [p != -1 for p in preds]
    valid_preds = [p for p, m in zip(preds, valid_mask) if m]
    valid_labels = [l for l, m in zip(labels, valid_mask) if m]
    valid_keywords = [k for k, m in zip(keywords, valid_mask) if m]

    n_invalid = sum(1 for p in preds if p == -1)
    logger.info(f"Valid responses: {len(valid_preds)}/{len(preds)} ({n_invalid} invalid/refused)")

    # Compute metrics
    metrics = compute_metrics(valid_labels, valid_preds)
    metrics["n_invalid"] = n_invalid
    metrics["n_total"] = len(preds)
    metrics["valid_rate"] = len(valid_preds) / len(preds)

    kw_metrics = per_keyword_metrics(valid_labels, valid_preds, valid_keywords)
    metrics["per_keyword"] = kw_metrics

    logger.info(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")

    # Save results
    results_dir = Path(cfg.paths.metrics_dir) / cfg.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save raw predictions for error analysis
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": preds,
        "raw_response": [r["raw_response"] for r in results],
        "is_valid": [r["is_valid"] for r in results],
        "keyword": keywords,
    })
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    logger.info("LLM evaluation complete!")


if __name__ == "__main__":
    main()
