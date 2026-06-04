#!/usr/bin/env python3
"""Per-checkpoint GPU forward-pass for Direction-1 CFR analysis.

For ONE (experiment, seed) checkpoint:
  1. Load model from ``$SCRATCH_ROOT/checkpoints/<exp>_seed<seed>/best_model.pt``
  2. Forward-pass test originals → ``test_originals.parquet``
  3. Forward-pass test swaps (read from shared manifest) → ``test_swap_logits.parquet``
  4. If exp is a BASELINE_METHOD (Model C), also forward-pass dev originals →
     ``dev_originals.parquet`` (needed by 06_apply_posthoc_baselines for fitting
     TS and multicalibration).
  5. Write ``manifest.json``.

Designed to run inside the RCP container (``$SCRATCH_ROOT == /scratch/case/...``).
Use ``analysis/direction1_rcp/submit_forward_pass.sh`` to fan out across all
checkpoints.

Re-running on the same checkpoint with the same swap manifest is a no-op
(manifest sha256 + swap_seed comparison).

Required env / inputs:
  * Shared swap manifest already built (run ``01_build_swap_manifest.py`` first)
  * Checkpoint at ``$SCRATCH_ROOT/checkpoints/<exp>_seed<seed>/best_model.pt``
  * Splits at ``data/processed/`` (relative to repo root)

Usage:
    python3 analysis/direction1/02_forward_pass.py --experiment davani_lambda10 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Repo root + this dir on path
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _utils import (  # noqa: E402
    BASELINE_METHODS,
    MULTIARCH_METHODS,
    SWEEP_METHODS,
    CacheManifest,
    cfr_cache_root,
    checkpoint_dir,
    file_sha256,
    load_model_state,
    read_manifest,
    write_manifest,
)
from src.models.transformer_classifier import TransformerClassifier  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


logger = get_logger(__name__)


def _resolve_config_for(experiment: str) -> str:
    """Pick a YAML config that matches the experiment's model architecture.

    All checkpoints we consider use ``microsoft/deberta-v3-base`` with
    dropout 0.1 (see configs/base.yaml + the per-experiment overrides), so
    any of these configs gives the right model + tokenizer at inference.
    We pick the one named after the experiment when available, else fall
    back to the canonical Model C config.
    """
    candidates = [
        f"configs/{experiment}.yaml",
        "configs/deberta_focal_mask.yaml",  # Model C — canonical fallback
    ]
    for c in candidates:
        if (_REPO_ROOT / c).exists():
            return c
    raise FileNotFoundError(
        f"No config matched for experiment={experiment!r}; tried {candidates}"
    )


def _build_model_for_inference(cfg) -> tuple[TransformerClassifier, "PreTrainedTokenizer"]:
    """Construct an inference-only :class:`TransformerClassifier`.

    The loss type is irrelevant at inference (no labels are passed in the
    forward), but :class:`TransformerClassifier`'s ``__init__`` requires
    ``loss_type`` so we pass ``"focal"`` as a benign default. Dropout
    matches base.yaml so the loaded state dict slots in cleanly.
    """
    model = TransformerClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        loss_type="focal",  # any value works — no labels at inference
        focal_gamma=2.0,
        focal_alpha=0.75,
    )
    tokenizer = TransformerClassifier.load_tokenizer(cfg.model.name)
    return model, tokenizer


@torch.no_grad()
def forward_pass_texts(
    model: TransformerClassifier,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Run ``model`` on ``texts`` in batches and stack ``(N, num_labels)`` logits.

    Returns float32 numpy. We deliberately do NOT use autocast: DeBERTa-v3
    overflows in fp16/bf16 (documented in the existing trainer.py); the
    tiny inference speedup is not worth the risk of NaN logits.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs["logits"].detach().cpu().float().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="GPU forward-pass for one Direction-1 checkpoint."
    )
    parser.add_argument("--experiment", required=True,
                        help=f"One of {SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS}")
    parser.add_argument("--seed", type=int, required=True,
                        help="One of {42,43,44,45,46,47}")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=0,
        help="Must match the seed used when building the swap manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the cache manifest matches.",
    )
    args = parser.parse_args()

    experiment = args.experiment
    seed = args.seed

    if experiment not in SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS:
        logger.error(
            "Unknown experiment %r. Allowed: %s",
            experiment, SWEEP_METHODS + BASELINE_METHODS + MULTIARCH_METHODS,
        )
        return 2

    # ------------------------------------------------------------------ paths
    ckpt_path = checkpoint_dir(experiment, seed) / "best_model.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 2

    cache_dir = cfr_cache_root() / f"{experiment}_seed{seed}"
    swap_manifest_path = cfr_cache_root() / "_shared" / "test_swaps_manifest.parquet"
    if not swap_manifest_path.exists():
        logger.error(
            "Shared swap manifest missing: %s. "
            "Run analysis/direction1/01_build_swap_manifest.py first.",
            swap_manifest_path,
        )
        return 2

    # ----------------------------------------------------------- short-circuit
    ckpt_sha = file_sha256(ckpt_path)
    existing = read_manifest(cache_dir)
    if (
        existing is not None
        and not args.force
        and existing.checkpoint_sha256 == ckpt_sha
        and existing.swap_engine_seed == args.swap_seed
    ):
        logger.info(
            "Cache up-to-date at %s (sha matches, swap_seed matches). Skipping.",
            cache_dir,
        )
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    config_rel = _resolve_config_for(experiment)
    cfg = load_config(_REPO_ROOT / config_rel)
    splits = load_splits(cfg.paths.processed_dir)
    test_df = splits["test"]

    text_col = cfg.data.text_column
    label_col = cfg.data.label_column
    keyword_col = cfg.data.keyword_column

    test_texts: list[str] = test_df[text_col].astype(str).tolist()
    test_labels: list[int] = test_df[label_col].astype(int).tolist()
    test_keywords: list[str] = test_df[keyword_col].astype(str).tolist()

    df_swaps = pd.read_parquet(swap_manifest_path)
    swap_texts: list[str] = df_swaps["swap_text"].astype(str).tolist()

    logger.info(
        "Forward-pass plan: %d test originals + %d swaps for %s seed=%d",
        len(test_texts), len(swap_texts), experiment, seed,
    )

    # ----------------------------------------------------------------- model
    seed_everything(seed)
    model, tokenizer = _build_model_for_inference(cfg)
    state = load_model_state(ckpt_path, device="cpu")
    missing_unexpected = model.load_state_dict(state, strict=True)
    # ``strict=True`` already raises on mismatch; this just logs success.
    logger.info("Loaded checkpoint %s", ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using GPU: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        logger.info("Using CPU (no CUDA)")
    model.to(device)

    # ------------------------------------------------------- forward originals
    logger.info("Forward-passing %d test originals…", len(test_texts))
    orig_logits = forward_pass_texts(
        model, tokenizer, test_texts, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    if orig_logits.shape != (len(test_texts), cfg.model.num_labels):
        raise RuntimeError(
            f"Expected logits shape ({len(test_texts)}, {cfg.model.num_labels}), "
            f"got {orig_logits.shape}"
        )

    orig_df = pd.DataFrame({
        "test_idx": np.arange(len(test_texts), dtype=np.int64),
        "text": test_texts,
        "true_label": np.asarray(test_labels, dtype=np.int64),
        "keyword": test_keywords,
        "logit_0": orig_logits[:, 0].astype(np.float32),
        "logit_1": orig_logits[:, 1].astype(np.float32),
    })
    orig_path = cache_dir / "test_originals.parquet"
    orig_df.to_parquet(orig_path, index=False)
    logger.info("Wrote %s", orig_path)

    # ----------------------------------------------------------- forward swaps
    if len(swap_texts) > 0:
        logger.info("Forward-passing %d test swaps…", len(swap_texts))
        swap_logits = forward_pass_texts(
            model, tokenizer, swap_texts, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        swap_out = df_swaps.copy()
        swap_out["logit_0"] = swap_logits[:, 0].astype(np.float32)
        swap_out["logit_1"] = swap_logits[:, 1].astype(np.float32)
        swap_path = cache_dir / "test_swap_logits.parquet"
        swap_out.to_parquet(swap_path, index=False)
        logger.info("Wrote %s", swap_path)
    else:
        logger.warning("Swap manifest is empty — no swap forward-pass to run.")

    # --------------------------------------------- forward dev (Model C only)
    if experiment in BASELINE_METHODS:
        dev_df = splits["dev"]
        dev_texts = dev_df[text_col].astype(str).tolist()
        dev_labels = dev_df[label_col].astype(int).tolist()
        dev_keywords = dev_df[keyword_col].astype(str).tolist()
        logger.info(
            "Baseline experiment %s: also forward-passing %d dev originals "
            "(needed for TS + multicalibration fitting)…",
            experiment, len(dev_texts),
        )
        dev_logits = forward_pass_texts(
            model, tokenizer, dev_texts, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        dev_out = pd.DataFrame({
            "dev_idx": np.arange(len(dev_texts), dtype=np.int64),
            "text": dev_texts,
            "true_label": np.asarray(dev_labels, dtype=np.int64),
            "keyword": dev_keywords,
            "logit_0": dev_logits[:, 0].astype(np.float32),
            "logit_1": dev_logits[:, 1].astype(np.float32),
        })
        dev_path = cache_dir / "dev_originals.parquet"
        dev_out.to_parquet(dev_path, index=False)
        logger.info("Wrote %s", dev_path)

    # -------------------------------------------------------------- manifest
    n_pairs_per_source = (
        df_swaps["source_group"].value_counts().to_dict() if len(df_swaps) else {}
    )
    manifest = CacheManifest(
        version=1,
        experiment=experiment,
        seed=seed,
        checkpoint_path=str(ckpt_path),
        checkpoint_sha256=ckpt_sha,
        swap_engine_seed=args.swap_seed,
        n_test_examples=len(test_texts),
        n_pairs_total=int(len(df_swaps)),
        n_pairs_per_group={str(k): int(v) for k, v in n_pairs_per_source.items()},
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_manifest(cache_dir, manifest)
    logger.info("Wrote manifest at %s", cache_dir / "manifest.json")
    logger.info("Done — Direction-1 forward-pass cache complete for %s seed=%d",
                experiment, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
