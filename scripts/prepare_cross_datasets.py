#!/usr/bin/env python3
"""Prepare Jewish-target cross-dataset CSVs for external evaluation.

This script downloads the source datasets from Hugging Face, extracts the
Jewish-target subsets, remaps labels to the binary format expected by the
project, and saves:

  - data/external/hatexplain_jewish.csv
  - data/external/toxigen_jewish.csv

Output schema for both files:
  - text: str
  - label: int  (1 = antisemitic / toxic, 0 = not antisemitic / benign)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import DatasetDict, load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


HATEXPLAIN_DATASET = "Hate-speech-CNERG/hatexplain"
TOXIGEN_DATASET = "toxigen/toxigen-data"
HATEXPLAIN_RAW_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"

# HateXplain annotator labels: 0 = hatespeech, 1 = normal, 2 = offensive
HATEXPLAIN_HATE_LABEL = 0
HATEXPLAIN_LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "hatespeech": 0,
    "hate speech": 0,
    "normal": 1,
    "offensive": 2,
}

JEWISH_TOKENS = {
    "jew",
    "jews",
    "jewish",
    "judaism",
}

TOXIGEN_JEWISH_GROUPS = {
    "jewish",
    "jews",
    "jew",
}


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def _contains_jewish_target(targets: Iterable[object]) -> bool:
    for target in targets:
        lowered = str(target).strip().lower()
        if lowered in JEWISH_TOKENS:
            return True
    return False


def _majority_label(labels: list[int]) -> int | None:
    """Strict majority; returns None on tie (prior tiebreak biased toward label 0=hatespeech)."""
    if not labels:
        return None
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda item: -item[1])
    if len(sorted_counts) >= 2 and sorted_counts[0][1] == sorted_counts[1][1]:
        return None  # ambiguous — drop rather than tiebreak
    return sorted_counts[0][0]


def _normalize_hatexplain_label(label: object) -> int | None:
    if label is None:
        return None
    if isinstance(label, int):
        return label
    lowered = str(label).strip().lower()
    return HATEXPLAIN_LABEL_MAP.get(lowered)


def prepare_hatexplain(output_path: Path) -> None:
    logger.info("Loading HateXplain from raw GitHub source...")
    with urlopen(HATEXPLAIN_RAW_URL) as response:
        raw_dataset = json.load(response)
    rows: list[dict[str, object]] = []
    logger.info("Processing HateXplain records...")
    for post_id, item in raw_dataset.items():
        annotators = item.get("annotators", []) or []
        labels = [
            normalized
            for ann in annotators
            for normalized in [_normalize_hatexplain_label(ann.get("label"))]
            if normalized is not None
        ]
        targets_nested = [ann.get("target", []) or [] for ann in annotators]
        flat_targets = [target for targets in targets_nested for target in targets]

        if not _contains_jewish_target(flat_targets):
            continue

        majority = _majority_label(labels)
        if majority is None:
            continue

        rows.append(
            {
                "text": _normalize_text(item.get("post_tokens", [])),
                "label": int(majority == HATEXPLAIN_HATE_LABEL),
                "post_id": post_id,
                "targets": json.dumps(sorted({str(t) for t in flat_targets})),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved HateXplain Jewish subset to {output_path} ({len(df)} rows)")


def prepare_toxigen(output_path: Path) -> None:
    logger.info("Loading ToxiGen from Hugging Face...")

    # Prefer the small human-annotated subset if available.
    try:
        dataset = load_dataset(TOXIGEN_DATASET, "annotated")
        text_column = "text"
        group_column = "target_group"
        label_column = "toxicity_human"
        logger.info("Using ToxiGen 'annotated' configuration")
    except Exception as exc:
        logger.warning(f"Annotated ToxiGen config unavailable ({exc}); falling back to default config")
        dataset = load_dataset(TOXIGEN_DATASET)
        text_column = "generation"
        group_column = "group"
        label_column = "prompt_label"

    rows: list[dict[str, object]] = []
    split_names = list(dataset.keys()) if isinstance(dataset, DatasetDict) else ["train"]

    for split_name in split_names:
        logger.info(f"Processing ToxiGen split: {split_name}")
        split = dataset[split_name]
        for item in split:
            group = str(item.get(group_column, "")).strip().lower()
            # Substring match: ToxiGen target_group is e.g. "jewish folks", not "jewish".
            if not any(tok in group for tok in TOXIGEN_JEWISH_GROUPS):
                continue

            raw_label = item.get(label_column)
            if raw_label is None:
                continue

            if isinstance(raw_label, str):
                try:
                    raw_label = float(raw_label)
                except ValueError:
                    continue

            label = int(float(raw_label) >= 3.0) if label_column == "toxicity_human" else int(raw_label)
            rows.append(
                {
                    "text": _normalize_text(item.get(text_column)),
                    "label": label,
                    "source_split": split_name,
                    "target_group": group,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved ToxiGen Jewish subset to {output_path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cross-dataset Jewish subsets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/external",
        help="Directory where the prepared CSV files should be saved",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    prepare_hatexplain(output_dir / "hatexplain_jewish.csv")
    prepare_toxigen(output_dir / "toxigen_jewish.csv")

    logger.info("Cross-dataset preparation complete.")


if __name__ == "__main__":
    main()
