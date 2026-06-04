#!/usr/bin/env python3
"""Build the shared counterfactual-swap manifest for the test set.

Deterministically generates Φ(x) for every test example using the existing
:class:`CounterfactualSwapEngine` with the heuristic symmetry filter, and
caches the result as a parquet file so that all 54 per-checkpoint
forward-pass jobs (script 02) can read the same swaps without racing on
generation.

Run ONCE before launching forward-pass jobs:

    python3 analysis/direction1/01_build_swap_manifest.py

The manifest goes to
``$SCRATCH_ROOT/cfr_cache/_shared/test_swaps_manifest.parquet``.

CPU-only. Idempotent (re-running with the same swap_engine_seed produces
the same file; pass --force to overwrite).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root + this dir on path
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _utils import (  # noqa: E402  (path setup must happen first)
    cfr_cache_root,
    normalise_keyword,
    resolve_scratch_root,
)
from src.data.counterfactual_swap import CounterfactualSwapEngine  # noqa: E402
from src.data.identity_inventory import primary_keyword_group  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


logger = get_logger(__name__)


def build_test_swap_manifest(
    test_texts: list[str],
    swap_engine_seed: int,
    swaps_per_token: int = 3,
    pairs_cap_per_example: int = 5,
) -> pd.DataFrame:
    """Generate (test_idx, swap_idx, swap_text, …) rows for every test sample.

    Uses :class:`HeuristicSymmetryClassifier` (the default of
    :class:`CounterfactualSwapEngine`), which rejects slur-bearing inputs
    and proxy-only inputs. Tweets without a swappable religious-ethnic
    Jewish token contribute zero rows — this is the by-design behaviour
    that makes ``kikes`` and ``zionazi`` groups have ``n_pairs = 0``.
    """
    rng = random.Random(swap_engine_seed)
    engine = CounterfactualSwapEngine(
        swaps_per_token=swaps_per_token,
        pairs_cap_per_example=pairs_cap_per_example,
        rng=rng,
    )

    rows: list[dict] = []
    for test_idx, text in enumerate(test_texts):
        source_group = primary_keyword_group(text)
        pairs = engine.generate(text)
        for swap_idx, p in enumerate(pairs):
            rows.append({
                "test_idx": test_idx,
                "swap_idx": swap_idx,
                "source_group": source_group,
                "original_text": text,
                "swap_text": p.swap_text,
                "original_token": p.original_token,
                "swap_token": p.swap_token,
                "alternative_group": p.alternative_group,
                "decision_reason": p.decision.reason,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build the shared test-set counterfactual-swap manifest."
    )
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=0,
        help="RNG seed for the swap engine (default: 0). Cached.",
    )
    parser.add_argument(
        "--swaps-per-token",
        type=int,
        default=3,
        help="K alternatives sampled per Jewish identity token (default: 3, "
             "matches training-time configs).",
    )
    parser.add_argument(
        "--pairs-cap",
        type=int,
        default=5,
        help="Max pairs retained per test example (default: 5, matches training).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deberta_focal_mask.yaml",
        help="Any base-config-inheriting YAML; used only to locate "
             "data/processed/ and the text/label/keyword column names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the manifest exists.",
    )
    args = parser.parse_args()

    cfg = load_config(_REPO_ROOT / args.config)
    splits = load_splits(cfg.paths.processed_dir)
    test_df = splits["test"]
    text_col = cfg.data.text_column

    out_dir = cfr_cache_root() / "_shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "test_swaps_manifest.parquet"
    meta_path = out_dir / "test_swaps_manifest_meta.json"

    if parquet_path.exists() and not args.force:
        logger.info(
            "Manifest already at %s. Pass --force to rebuild. Exiting.",
            parquet_path,
        )
        return 0

    logger.info(
        "Building swap manifest for %d test examples (swap_seed=%d, K=%d, cap=%d)…",
        len(test_df), args.swap_seed, args.swaps_per_token, args.pairs_cap,
    )
    df = build_test_swap_manifest(
        test_texts=test_df[text_col].astype(str).tolist(),
        swap_engine_seed=args.swap_seed,
        swaps_per_token=args.swaps_per_token,
        pairs_cap_per_example=args.pairs_cap,
    )

    # Also report identity-group composition of the test set (for audit)
    keyword_col = cfg.data.keyword_column
    test_keyword_counts = (
        test_df[keyword_col].astype(str).map(normalise_keyword).value_counts().to_dict()
    )

    df.to_parquet(parquet_path, index=False)

    n_pairs_per_source = df["source_group"].value_counts().to_dict()
    n_pairs_per_alt = df["alternative_group"].value_counts().to_dict()
    n_examples_with_pairs = int(df["test_idx"].nunique()) if len(df) else 0

    meta = {
        "swap_engine_seed": args.swap_seed,
        "swaps_per_token": args.swaps_per_token,
        "pairs_cap_per_example": args.pairs_cap,
        "config_used": args.config,
        "n_test_examples": int(len(test_df)),
        "n_test_examples_with_pairs": n_examples_with_pairs,
        "n_pairs_total": int(len(df)),
        "n_pairs_per_source_group": {str(k): int(v) for k, v in n_pairs_per_source.items()},
        "n_pairs_per_alternative_group": {str(k): int(v) for k, v in n_pairs_per_alt.items()},
        "test_keyword_distribution": {str(k): int(v) for k, v in test_keyword_counts.items()},
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote %s (%d rows) and %s", parquet_path, len(df), meta_path)
    logger.info(
        "Coverage: %d/%d test examples produced >=1 symmetric pair (%.1f%%)",
        n_examples_with_pairs, len(test_df), 100 * n_examples_with_pairs / max(1, len(test_df)),
    )
    logger.info("Pairs per source group: %s", n_pairs_per_source)
    logger.info("Pairs per alternative group: %s", n_pairs_per_alt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
