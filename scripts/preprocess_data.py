#!/usr/bin/env python3
"""Preprocess GoldStandard2024 and create train/dev/test splits.

Run this once before training:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

# Allow `python scripts/preprocess_data.py` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.data.preprocessing import preprocess_tweet
from src.data.splits import create_splits, save_splits

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data and create splits")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    processed_dir = Path(cfg.paths.processed_dir)
    if (processed_dir / "train.csv").exists() and not args.force:
        logger.info("Splits already exist. Use --force to overwrite.")
        return

    # Load raw data
    raw_path = Path(cfg.paths.raw_data)
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run data/download.sh first.")
        return

    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} samples")

    # Show dataset stats
    logger.info(f"Label distribution:\n{df[cfg.data.label_column].value_counts()}")
    logger.info(f"Keyword distribution:\n{df[cfg.data.keyword_column].value_counts()}")

    # Preprocess text
    logger.info("Preprocessing tweets...")
    df[cfg.data.text_column] = df[cfg.data.text_column].apply(
        lambda x: preprocess_tweet(
            x,
            mask_usernames=cfg.preprocessing.mask_usernames,
            remove_urls=cfg.preprocessing.remove_urls,
            remove_hashtag_symbol=cfg.preprocessing.remove_hashtag_symbol,
            lowercase=cfg.preprocessing.lowercase,
        )
    )

    # Show sample preprocessed texts
    logger.info("Sample preprocessed texts:")
    for i in range(min(3, len(df))):
        logger.info(f"  [{df[cfg.data.label_column].iloc[i]}] {df[cfg.data.text_column].iloc[i][:100]}...")

    # Create stratified splits
    logger.info("Creating stratified splits...")
    splits = create_splits(
        df,
        label_col=cfg.data.label_column,
        keyword_col=cfg.data.keyword_column,
        train_ratio=cfg.data.train_ratio,
        dev_ratio=cfg.data.dev_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )

    # Save
    save_splits(splits, cfg.paths.processed_dir)

    # Verify splits: total may be < len(df) if duplicate texts were deduped
    total = sum(len(s) for s in splits.values())
    logger.info(f"Total samples across splits: {total} (raw loaded: {len(df)})")
    assert total <= len(df), "Split sizes exceed input size!"

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
