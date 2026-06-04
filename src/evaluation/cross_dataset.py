"""Cross-dataset zero-shot evaluation."""
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AntisemitismDataset
from src.evaluation.metrics import compute_metrics
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_hatexplain_jewish_subset(data_dir: str) -> pd.DataFrame:
    """
    Load HateXplain and filter to Jewish-target posts.

    HateXplain has columns: post_id, text, label (hate/offensive/normal),
    target (list of communities).

    We map: hate+target=Jews → antisemitic (1), else → not antisemitic (0)
    """
    path = Path(data_dir) / "hatexplain_jewish.csv"

    if not path.exists():
        logger.warning(f"HateXplain Jewish subset not found at {path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded HateXplain Jewish subset: {len(df)} samples")
    return df


def load_toxigen_jewish_subset(data_dir: str) -> pd.DataFrame:
    """
    Load ToxiGen and filter to Jewish group.

    ToxiGen has columns: text, target_group, toxicity_label
    We map: toxic + target=Jewish → 1, else → 0
    """
    path = Path(data_dir) / "toxigen_jewish.csv"

    if not path.exists():
        logger.warning(f"ToxiGen Jewish subset not found at {path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded ToxiGen Jewish subset: {len(df)} samples")
    return df


def cross_dataset_evaluate(
    model,
    tokenizer,
    dataset_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 128,
    batch_size: int = 32,
    device: str = "cuda",
    predictions_out_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Run zero-shot evaluation of our model on an external dataset.

    If `predictions_out_path` is provided, also writes a CSV with one row per
    example containing (text, label, prob_positive, pred_at_0.5). This is what
    the Tier-1A threshold-sweep aggregator (scripts/threshold_sweep_cross_dataset.py)
    consumes — it lets us re-score the same predictions at PR-curve-optimal
    thresholds without re-running inference.
    """
    if dataset_df.empty:
        return {}

    dataset = AntisemitismDataset(
        texts=dataset_df[text_col].tolist(),
        labels=dataset_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = outputs["logits"].argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    if predictions_out_path is not None:
        out_path = Path(predictions_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "text": dataset_df[text_col].tolist(),
            "label": all_labels,
            "prob_positive": all_probs,
            "pred_at_0.5": all_preds,
        }).to_csv(out_path, index=False)
        logger.info(f"Per-example predictions written to {out_path}")

    return compute_metrics(all_labels, all_preds, all_probs)
