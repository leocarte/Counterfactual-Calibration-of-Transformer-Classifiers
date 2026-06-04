"""Stratified data splitting with joint Biased × Keyword stratification."""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_splits(
    df: pd.DataFrame,
    label_col: str = "Biased",
    keyword_col: str = "Keyword",
    text_col: str = "Text",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dedupe_text: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Create train/dev/test splits stratified by label × keyword.

    Joint stratification ensures each keyword group has proportional
    representation of antisemitic/non-antisemitic tweets in all splits.

    When dedupe_text=True (default), rows sharing identical Text are collapsed
    to a single row before stratification. This prevents the same tweet from
    appearing in both train and test, which would inflate F1 through
    memorization. GoldStandard2024 contains ~421 duplicated texts (1088 rows)
    from tweets appearing under multiple Keyword buckets.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    df = df.copy()

    if dedupe_text and text_col in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                f"Deduped {n_dropped} duplicate texts "
                f"({n_before} -> {len(df)} rows) to prevent cross-split leakage"
            )

    # Create joint stratification key
    df["_strat_key"] = df[label_col].astype(str) + "_" + df[keyword_col].astype(str)

    # First split: train+dev vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_dev_idx, test_idx = next(sss1.split(df, df["_strat_key"]))

    # Second split: train vs dev (from train+dev portion)
    dev_fraction = dev_ratio / (train_ratio + dev_ratio)
    df_train_dev = df.iloc[train_dev_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=seed)
    train_idx_rel, dev_idx_rel = next(sss2.split(df_train_dev, df_train_dev["_strat_key"]))

    train_idx = train_dev_idx[train_idx_rel]
    dev_idx = train_dev_idx[dev_idx_rel]

    splits = {
        "train": df.iloc[train_idx].drop(columns=["_strat_key"]).reset_index(drop=True),
        "dev": df.iloc[dev_idx].drop(columns=["_strat_key"]).reset_index(drop=True),
        "test": df.iloc[test_idx].drop(columns=["_strat_key"]).reset_index(drop=True),
    }

    # Log split statistics
    for name, split_df in splits.items():
        n_pos = split_df[label_col].sum()
        n_total = len(split_df)
        logger.info(f"{name}: {n_total} samples, {n_pos} positive ({n_pos/n_total:.1%})")

    return splits


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save splits as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(output_path / f"{name}.csv", index=False)
        logger.info(f"Saved {name} split to {output_path / f'{name}.csv'}")


def load_splits(processed_dir: str) -> dict[str, pd.DataFrame]:
    """Load previously saved splits."""
    processed_path = Path(processed_dir)
    return {
        name: pd.read_csv(processed_path / f"{name}.csv")
        for name in ["train", "dev", "test"]
    }
