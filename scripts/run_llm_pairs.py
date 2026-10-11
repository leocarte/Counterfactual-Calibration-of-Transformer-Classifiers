#!/usr/bin/env python3
"""Run LLM inference on BOTH sides of the Direction-1 counterfactual-pair
manifest and compute the counterfactual flip rate (CFR_hard) per model.

Encoder checkpoints get their CFR from cached logits
(``analysis/direction1/02_forward_pass.py`` + ``03_compute_cfr.py``). The LLM
prompting baselines (D1/D2/D3) expose no logits, so instead we prompt the
model on the original tweet x AND its identity-swapped counterfactual x'
(same prompt template on both sides, mirroring how the same encoder is
applied to both sides) and count hard label flips over pairs.

Unlike encoder logits, LLM outputs are not always parseable, so we report
coverage explicitly and bracket the full-denominator CFR:

  * ``coverage_pairs_both``       = pairs parseable on BOTH sides / n_pairs
  * ``cfr_hard``                  = flips / pairs parseable on both sides
                                    (the flip rate over covered pairs — primary)
  * ``cfr_hard_lower_full_denom`` = flips / n_pairs
                                    (unparseable pairs counted as non-flips)
  * ``cfr_hard_upper_full_denom`` = (flips + unparseable pairs) / n_pairs
                                    (unparseable pairs counted as flips)

The lower/upper bounds bracket what CFR_hard would be on the encoder
denominator (2,117 pairs), keeping LLM rows comparable to the main table.

Decoding is greedy (temperature 0.0 in all three model configs), so identical
texts produce identical outputs — originals repeated across pairs are
generated ONCE and joined back per pair (~2.7k generations instead of
2 x 2,117). Every generation streams to an append-only JSONL cache in the
output dir; re-running the same command RESUMES instead of regenerating
(a killed RCP job loses at most --chunk-size generations).

Usage (inside the RCP container; see rcp/submit_llm_cfr.sh):

    python3 scripts/run_llm_pairs.py --model-config configs/llm_guidedcot.yaml
    python3 scripts/run_llm_pairs.py --model-config configs/qwen_guidedcot.yaml \
        --prompt-strategy guided_cot --limit 25      # smoke test

Outputs (default under $SCRATCH_ROOT/llm_cfr/<experiment>__<strategy>/):
    pair_predictions.csv       one row per manifest pair, predictions + raw text
    cfr_summary.json           CFR_hard + coverage + per-group breakdown
    generations_cache.jsonl    append-only text->generation cache (resume)

Raw responses are persisted in pair_predictions.csv, so a more tolerant
parser can be re-applied offline later WITHOUT re-running any GPU job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/run_llm_pairs.py ...` without PYTHONPATH=. ; the
# Direction-1 dir is added for its _utils helpers (scratch-root resolution,
# file hashing) — same path-setup pattern as the numbered scripts there.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(1, str(_REPO_ROOT / "analysis" / "direction1"))

import numpy as np
import pandas as pd

from _utils import file_sha256, resolve_scratch_root  # noqa: E402
from src.data.identity_inventory import PRIMARY_KEYWORD_GROUPS  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.models.llm_prompter import LLMPrompter  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

logger = get_logger(__name__)

#: Same mapping as scripts/run_llm.py — keep the two scripts consistent.
LABEL_MAP = {"antisemitic": 1, "not antisemitic": 0}
INVALID = -1

#: Template names under src/models/prompts/<name>.txt
PROMPT_STRATEGIES = ("zero_shot", "few_shot", "guided_cot")


# ---------------------------------------------------------------------------
# Generation cache (append-only JSONL keyed by sha256 of the input text).
# Model + prompt strategy are fixed per output dir, so the text alone is a
# sufficient key. Greedy decoding makes the cache exact, not approximate.
# ---------------------------------------------------------------------------

def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_generation_cache(path: Path) -> dict[str, dict]:
    """Read the JSONL cache, dropping torn trailing writes from killed jobs."""
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    n_bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[str(rec["key"])] = rec
            except (json.JSONDecodeError, KeyError, TypeError):
                n_bad += 1
    if n_bad:
        logger.warning(
            "Generation cache had %d unreadable line(s) (torn write from a "
            "killed job?) — those texts will be regenerated.", n_bad,
        )
    logger.info("Loaded %d cached generation(s) from %s", len(cache), path)
    return cache


def _classify_with_cache(
    prompter: LLMPrompter,
    texts: list[str],
    cache: dict[str, dict],
    cache_path: Path,
    chunk_size: int,
    side: str,
) -> int:
    """Generate predictions for every text in ``texts`` not already cached.

    Updates ``cache`` in place and appends each chunk to ``cache_path``
    (flushed per chunk, so a killed job loses at most ``chunk_size``
    generations). Returns the number of NEW generations run.
    """
    pending: list[tuple[str, str]] = []
    uniq: set[str] = set()
    for t in texts:
        k = _text_key(t)
        if k in uniq:
            continue
        uniq.add(k)
        if k not in cache:
            pending.append((k, t))

    logger.info(
        "[%s] %d unique text(s): %d cached, %d to generate",
        side, len(uniq), len(uniq) - len(pending), len(pending),
    )
    if not pending:
        return 0

    with cache_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            results = prompter.classify_batch([t for _, t in chunk])
            for (k, _t), r in zip(chunk, results):
                rec = {
                    "key": k,
                    "prediction": r["prediction"],
                    "raw_response": r["raw_response"],
                }
                cache[k] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            logger.info(
                "[%s] generated %d/%d (cache flushed)",
                side, min(start + chunk_size, len(pending)), len(pending),
            )
    return len(pending)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_group_breakdown(pair_df: pd.DataFrame) -> dict[str, dict]:
    """CFR_hard + coverage per source group, in PRIMARY_KEYWORD_GROUPS order.

    Mirrors the per-group structure of 03_compute_cfr.py. Groups absent from
    the manifest (kikes / zionazi have 0 pairs by construction of the
    symmetry filter) appear with n_pairs = 0 so the audit trail is explicit.
    """
    src = pair_df["source_group"].astype(str)
    groups = list(PRIMARY_KEYWORD_GROUPS) + sorted(
        set(src) - set(PRIMARY_KEYWORD_GROUPS)
    )
    out: dict[str, dict] = {}
    for g in groups:
        sub = pair_df[src == g]
        n_g = int(len(sub))
        n_both = int(sub["parseable_both"].sum())
        n_flips = int((sub["flip"].to_numpy() == 1).sum())
        out[g] = {
            "n_pairs": n_g,
            "n_pairs_parseable_both": n_both,
            "coverage_pairs_both": (n_both / n_g) if n_g else None,
            "n_flips": n_flips,
            "cfr_hard": (n_flips / n_both) if n_both else None,
        }
    return out


def _unique_parse_stats(texts: list[str], cache: dict[str, dict]) -> tuple[int, float | None]:
    """(n_unique_texts, fraction of unique texts whose output parsed)."""
    seen: dict[str, bool] = {}
    for t in texts:
        k = _text_key(t)
        if k not in seen:
            seen[k] = LABEL_MAP.get(cache[k]["prediction"], INVALID) != INVALID
    n = len(seen)
    return n, (sum(seen.values()) / n if n else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM counterfactual flip rate (CFR_hard) over the "
                    "Direction-1 pair manifest",
    )
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Model YAML (e.g. configs/llm_guidedcot.yaml, "
             "configs/mistral_guidedcot.yaml, configs/qwen_guidedcot.yaml)",
    )
    parser.add_argument(
        "--manifest-path", type=str, default=None,
        help="Parquet pair manifest. Default: "
             "<scratch>/cfr_cache/_shared/test_swaps_manifest.parquet "
             "(container: /scratch/case/antisemitism-results/...)",
    )
    parser.add_argument(
        "--prompt-strategy", type=str, default=None,
        choices=list(PROMPT_STRATEGIES),
        help="Prompt template name (src/models/prompts/<name>.txt). "
             "Default: cfg.llm.prompt_template from the model config.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Default: <scratch>/llm_cfr/<experiment_name>__<strategy>"
             "[__limitN]",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Use only the first N manifest pairs (smoke tests).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=25,
        help="Generations per cache flush — the resume granularity "
             "(default: 25, roughly one flush every 2-4 min of CoT decoding).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Override cfg.llm.max_new_tokens. A zero-shot CFR sweep only needs "
             "a short label, so a low cap (e.g. 64) avoids the CoT-sized 300 "
             "tokens that the guided-CoT model config carries.",
    )
    args = parser.parse_args()

    cfg = load_config(args.model_config)
    seed_everything(cfg.seed)

    prompt_strategy = args.prompt_strategy or str(cfg.llm.prompt_template)

    temperature = float(cfg.llm.temperature)
    if temperature != 0.0:
        logger.warning(
            "cfg.llm.temperature=%.2f != 0 — decoding is stochastic; CFR_hard "
            "will not be reproducible and the per-text generation cache "
            "becomes approximate. The CFR protocol assumes greedy decoding.",
            temperature,
        )

    # ----------------------------------------------------------- input paths
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = (
            resolve_scratch_root() / "cfr_cache" / "_shared"
            / "test_swaps_manifest.parquet"
        )
    if not manifest_path.exists():
        logger.error(
            "Pair manifest not found: %s\n"
            "It lives on the RCP group scratch PVC (in-container "
            "/scratch/case/antisemitism-results/cfr_cache/_shared/, jumphost "
            "/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/case/...). "
            "Build it once on the jumphost with "
            "`python3 analysis/direction1/01_build_swap_manifest.py`, or pass "
            "--manifest-path explicitly.",
            manifest_path,
        )
        return 2

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        leaf = f"{cfg.experiment_name}__{prompt_strategy}"
        if args.limit is not None:
            # Keep smoke runs out of the full-run directory so a later full
            # run never clobbers (or gets clobbered by) a smoke summary.
            leaf += f"__limit{args.limit}"
        output_dir = resolve_scratch_root() / "llm_cfr" / leaf
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # -------------------------------------------------------------- manifest
    manifest_sha = file_sha256(manifest_path)
    pairs = pd.read_parquet(manifest_path)
    n_pairs_manifest = int(len(pairs))
    logger.info(
        "Manifest: %s (%d pairs, sha256=%s)",
        manifest_path, n_pairs_manifest, manifest_sha[:12],
    )
    if args.limit is not None and args.limit < len(pairs):
        pairs = pairs.head(args.limit).reset_index(drop=True)
        logger.info(
            "LIMIT active: using first %d/%d manifest pairs (smoke-test mode)",
            len(pairs), n_pairs_manifest,
        )
    if len(pairs) == 0:
        logger.error("Manifest selection is empty — nothing to do.")
        return 2

    # True labels for the original side (sanity / side-quality metrics only —
    # CFR itself never uses labels). test_idx indexes the test split order,
    # exactly as in 02_forward_pass.py.
    splits = load_splits(cfg.paths.processed_dir)
    test_df = splits["test"]
    text_col = cfg.data.text_column
    label_col = cfg.data.label_column
    max_idx = int(pairs["test_idx"].max())
    if max_idx >= len(test_df):
        logger.error(
            "Manifest test_idx max=%d exceeds test split size %d — "
            "data/processed/ is out of sync with the manifest. Do NOT "
            "regenerate either side silently; investigate first.",
            max_idx, len(test_df),
        )
        return 2
    test_texts = test_df[text_col].astype(str).to_numpy()
    test_labels = test_df[label_col].astype(int).to_numpy()
    idx = pairs["test_idx"].to_numpy(np.int64)
    n_text_mismatch = int(
        (test_texts[idx] != pairs["original_text"].astype(str).to_numpy()).sum()
    )
    if n_text_mismatch:
        logger.warning(
            "%d/%d pairs: manifest original_text differs from "
            "data/processed/test.csv — true-label side metrics may be "
            "unreliable (CFR itself is unaffected; it only uses the texts "
            "stored in the manifest).",
            n_text_mismatch, len(pairs),
        )
    pairs = pairs.copy()
    pairs["true_label"] = test_labels[idx]

    logger.info(
        "Pairs per source group: %s",
        pairs["source_group"].value_counts().to_dict(),
    )

    # ----------------------------------------------------------------- model
    prompter = LLMPrompter(
        model_name=cfg.llm.model_name,
        prompt_template=prompt_strategy,
        quantization=cfg.llm.quantization,
        max_new_tokens=(args.max_new_tokens if args.max_new_tokens is not None else cfg.llm.max_new_tokens),
        temperature=temperature,
    )

    cache_path = output_dir / "generations_cache.jsonl"
    cache = _load_generation_cache(cache_path)

    original_texts = pairs["original_text"].astype(str).tolist()
    swap_texts = pairs["swap_text"].astype(str).tolist()

    t0 = time.time()
    n_new = 0
    n_new += _classify_with_cache(
        prompter, original_texts, cache, cache_path, args.chunk_size,
        side="originals",
    )
    n_new += _classify_with_cache(
        prompter, swap_texts, cache, cache_path, args.chunk_size,
        side="swaps",
    )
    gen_seconds = time.time() - t0

    # ------------------------------------------------------------ join pairs
    def _rec(text: str) -> dict:
        return cache[_text_key(text)]

    pred_orig_str = [_rec(t)["prediction"] for t in original_texts]
    pred_swap_str = [_rec(t)["prediction"] for t in swap_texts]
    pred_orig = np.array(
        [LABEL_MAP.get(p, INVALID) for p in pred_orig_str], dtype=np.int64,
    )
    pred_swap = np.array(
        [LABEL_MAP.get(p, INVALID) for p in pred_swap_str], dtype=np.int64,
    )
    both = (pred_orig != INVALID) & (pred_swap != INVALID)
    # flip: 1 = label flipped, 0 = stable, -1 = not parseable on both sides
    flip = np.where(both, (pred_orig != pred_swap).astype(np.int64), INVALID)

    pair_out = pairs.copy()
    pair_out["pred_original"] = pred_orig_str
    pair_out["pred_swap"] = pred_swap_str
    pair_out["pred_original_int"] = pred_orig
    pair_out["pred_swap_int"] = pred_swap
    pair_out["parseable_both"] = both
    pair_out["flip"] = flip
    pair_out["raw_response_original"] = [
        _rec(t)["raw_response"] for t in original_texts
    ]
    pair_out["raw_response_swap"] = [_rec(t)["raw_response"] for t in swap_texts]

    front = [
        "test_idx", "swap_idx", "source_group", "alternative_group",
        "original_token", "swap_token", "decision_reason", "true_label",
        "pred_original", "pred_swap", "pred_original_int", "pred_swap_int",
        "parseable_both", "flip",
        "original_text", "swap_text",
        "raw_response_original", "raw_response_swap",
    ]
    cols = [c for c in front if c in pair_out.columns]
    cols += [c for c in pair_out.columns if c not in cols]
    pair_csv = output_dir / "pair_predictions.csv"
    pair_out[cols].to_csv(pair_csv, index=False)
    logger.info("Wrote %s (%d rows)", pair_csv, len(pair_out))

    # ----------------------------------------------------------- aggregation
    n_pairs = int(len(pair_out))
    n_both = int(both.sum())
    n_unparseable = n_pairs - n_both
    n_flips = int((flip == 1).sum())
    cfr_hard = (n_flips / n_both) if n_both else None
    coverage = n_both / n_pairs

    n_uniq_orig, parse_rate_uniq_orig = _unique_parse_stats(original_texts, cache)
    n_uniq_swap, parse_rate_uniq_swap = _unique_parse_stats(swap_texts, cache)
    n_uniq_union = len({_text_key(t) for t in original_texts + swap_texts})

    per_group = _per_group_breakdown(pair_out)

    # Side quality on covered originals: accuracy + F1+ vs true labels, on
    # unique originals whose output parsed. Same formula as
    # 03_compute_cfr.py::_classification_metrics_at_threshold. NOT comparable
    # to the full-test-set F1 from scripts/run_llm.py (this is the with-pairs
    # subset only) — it is a consistency anchor, not a headline number.
    dedup = pair_out.drop_duplicates(subset="test_idx")
    valid = dedup[dedup["pred_original_int"] != INVALID]
    if len(valid):
        y_true = valid["true_label"].to_numpy(np.int64)
        y_pred = valid["pred_original_int"].to_numpy(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        denom = 2 * tp + fp + fn
        f1_pos = (2 * tp / denom) if denom else None
        acc = float((y_pred == y_true).mean())
    else:
        f1_pos, acc = None, None

    summary = {
        "experiment_name": str(cfg.experiment_name),
        "model_name": str(cfg.llm.model_name),
        "model_config": str(args.model_config),
        "prompt_strategy": prompt_strategy,
        "quantization": str(cfg.llm.quantization),
        "max_new_tokens": int(cfg.llm.max_new_tokens),
        "temperature": temperature,
        "greedy_decoding": temperature == 0.0,
        "seed": int(cfg.seed),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "n_pairs_manifest_full": n_pairs_manifest,
        "limit": args.limit,
        "n_pairs_used": n_pairs,
        "n_pairs_per_source_group": {
            str(k): int(v)
            for k, v in pair_out["source_group"].value_counts().items()
        },
        "n_unique_original_texts": n_uniq_orig,
        "n_unique_swap_texts": n_uniq_swap,
        "n_unique_texts_union": n_uniq_union,
        "parse": {
            "n_pairs_original_side_parseable": int((pred_orig != INVALID).sum()),
            "n_pairs_swap_side_parseable": int((pred_swap != INVALID).sum()),
            "n_pairs_parseable_both": n_both,
            "n_pairs_unparseable_any_side": n_unparseable,
            "coverage_pairs_both": coverage,
            "parse_rate_unique_originals": parse_rate_uniq_orig,
            "parse_rate_unique_swaps": parse_rate_uniq_swap,
        },
        "cfr": {
            "n_flips": n_flips,
            "cfr_hard": cfr_hard,
            "cfr_hard_lower_full_denom": n_flips / n_pairs,
            "cfr_hard_upper_full_denom": (n_flips + n_unparseable) / n_pairs,
            "per_group": per_group,
        },
        "metric_definitions": {
            "cfr_hard": "flips / pairs parseable on both sides (primary; "
                        "flip rate over covered pairs)",
            "coverage_pairs_both": "pairs parseable on both sides / n_pairs_used",
            "cfr_hard_lower_full_denom": "flips / n_pairs_used "
                                         "(unparseable pairs counted as non-flips)",
            "cfr_hard_upper_full_denom": "(flips + unparseable pairs) / "
                                         "n_pairs_used (unparseable counted "
                                         "as flips)",
        },
        "covered_originals_quality": {
            "n_unique_originals_parseable": int(len(valid)),
            "accuracy": acc,
            "f1_positive": f1_pos,
            "n_manifest_text_mismatches_vs_test_split": n_text_mismatch,
            "note": "with-pairs subset only — not comparable to the "
                    "full-test-set F1 from scripts/run_llm.py",
        },
        "runtime": {
            "generation_seconds": round(gen_seconds, 1),
            "n_generations_new": int(n_new),
            "n_generations_from_cache": int(n_uniq_union - n_new),
            "seconds_per_new_generation": (
                round(gen_seconds / n_new, 2) if n_new else None
            ),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / "cfr_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote %s", summary_path)

    cfr_fmt = f"{cfr_hard:.4f}" if cfr_hard is not None else "n/a"
    logger.info(
        "[%s | %s] CFR_hard=%s over %d/%d parseable-both pairs "
        "(coverage=%.3f, flips=%d, bounds on full denom: %.4f-%.4f)",
        cfg.experiment_name, prompt_strategy, cfr_fmt, n_both, n_pairs,
        coverage, n_flips,
        n_flips / n_pairs, (n_flips + n_unparseable) / n_pairs,
    )
    logger.info("LLM pair sweep complete! Outputs in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
