"""Compute Integrated Gradients attribution maps for antisemitism classification.

Per-token attribution via Captum, summed over the embedding dimension. For each
test example we also compute the fraction of |attribution| mass that lands on
identity tokens (jew*, israel*, zion*, kike*) — the headline metric for §4.5.

Handles checkpoints from all three training paths:
  * Model B (vanilla DeBERTa-v3-base)        — flat state dict
  * Model C (focal + identity-mask)          — flat state dict
  * CCI v2 (fixed_js, learned_js, ...)       — wrapped {"model", "temperature_head"}

Usage:
    python scripts/eval_attribution.py \
        --checkpoint /scratch/case/antisemitism-results/checkpoints/<dir>/best_model.pt \
        --tag <name> --seed <int> \
        --output results/interpretability/<tag>_s<seed>_ig.json \
        --n-examples 500 [--n-steps 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer

from src.models.transformer_classifier import TransformerClassifier
from src.data.identity_inventory import find_identity_matches
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_checkpoint(path: str, model_name: str, device: str) -> TransformerClassifier:
    """Load checkpoint. Unwraps CCI v2 dict ({"model", "temperature_head"}).

    strict=False: loss_fn buffers can be missing depending on train-time loss_type.
    >10 missing/unexpected = abort (catches wrong --model-name silently masking arch mismatch).
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state and "temperature_head" in state:
        logger.info("Detected wrapped CCI v2 checkpoint; loading state['model']")
        state = state["model"]

    model = TransformerClassifier(
        model_name=model_name,
        num_labels=2,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"State loaded. Missing keys: {len(missing)} (loss_fn buffers expected; up to ~5). "
        f"Unexpected keys: {len(unexpected)}"
    )
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            f"Checkpoint/model architecture mismatch suspected: missing={len(missing)} "
            f"unexpected={len(unexpected)}. The --model-name '{model_name}' may not "
            f"match the checkpoint. First 5 missing: {missing[:5]}. "
            f"First 5 unexpected: {unexpected[:5]}."
        )
    model.eval()
    return model


def get_attributions(
    model: TransformerClassifier,
    tokenizer: AutoTokenizer,
    text: str,
    device: str,
    n_steps: int = 50,
) -> tuple[list[str], list[float], float]:
    """Run Integrated Gradients on a single example.

    Returns: (tokens, per-token attribution sums over embedding dim, predicted P(positive)).
    """
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    # TransformerClassifier wraps AutoModelForSequenceClassification as self.model.
    # Use the HF-standard accessor to get the embedding layer.
    embed_layer = model.model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)

    def forward_for_ig(input_embeds_, attention_mask_):
        # AutoModelForSequenceClassification.forward() accepts inputs_embeds
        # directly. Outputs.logits has shape (B, num_labels).
        outputs = model.model(
            inputs_embeds=input_embeds_,
            attention_mask=attention_mask_,
        )
        return outputs.logits[:, 1]  # positive-class logit

    ig = IntegratedGradients(forward_for_ig)
    attributions = ig.attribute(
        inputs=input_embeds,
        baselines=torch.zeros_like(input_embeds),
        additional_forward_args=(attn_mask,),
        n_steps=n_steps,
    )
    # Sum over embedding dim → per-token attribution. Shape (seq_len,).
    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

    # Softmax over both logits (not sigmoid on logit[:,1]) so pred_prob is
    # unbiased near the boundary. IG target is unaffected.
    with torch.no_grad():
        full_logits = model.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
        ).logits
        prob_pos = torch.softmax(full_logits, dim=-1)[0, 1].item()

    return tokens, token_attr.tolist(), prob_pos


def fraction_on_identity(text: str, tokens: list[str], attributions: list[float]) -> float:
    """Fraction of |attribution| mass on identity tokens (jew*, israel*, zion*, kike*)."""
    matches = find_identity_matches(text)
    spans = [(m.start, m.end) for m in matches]
    if not spans:
        return 0.0

    abs_attr = np.abs(attributions)
    total = float(abs_attr.sum())
    if total < 1e-8:
        return 0.0

    identity_token_indices = []
    char_pos = 0
    for i, tok in enumerate(tokens):
        # Strip subword prefixes: ▁ (SentencePiece), Ġ (BPE/RoBERTa), ## (WordPiece).
        # removeprefix, not lstrip — lstrip strips a char-set.
        clean = (
            tok.removeprefix("▁")
               .removeprefix("Ġ")
               .removeprefix("##")
               .lower()
        )
        if not clean or tok in {
            "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
            "<s>", "</s>", "<pad>", "<unk>", "<mask>",
        }:
            continue
        idx = text.lower().find(clean, char_pos)
        if idx == -1:
            continue
        tok_end = idx + len(clean)
        for s, e in spans:
            if not (tok_end <= s or idx >= e):
                identity_token_indices.append(i)
                break
        char_pos = tok_end

    if not identity_token_indices:
        return 0.0
    return float(abs_attr[identity_token_indices].sum() / total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--tag", required=True, help="Short label for this run (e.g. 'modelb', 'cciv2')")
    ap.add_argument("--seed", type=int, required=True, help="Seed of the source checkpoint (for filename)")
    ap.add_argument("--output", required=True, help="Path to output JSON")
    ap.add_argument("--n-examples", type=int, default=500, help="Number of test examples (default 500)")
    ap.add_argument("--n-steps", type=int, default=50, help="IG Riemann steps (default 50)")
    ap.add_argument(
        "--model-name", default="microsoft/deberta-v3-base",
        help="HF model identifier — must match the architecture used to train the "
             "checkpoint. Examples: 'microsoft/deberta-v3-base' (default), "
             "'microsoft/deberta-v3-large', 'roberta-large', 'roberta-base'.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}; model_name: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_checkpoint(args.checkpoint, args.model_name, device)

    test_df = pd.read_csv("data/processed/test.csv").head(args.n_examples)
    logger.info(f"Running IG on {len(test_df)} examples × {args.n_steps} steps")

    results = []
    skipped = 0
    for i, row in test_df.iterrows():
        text = str(row["Text"])
        try:
            toks, attr, prob = get_attributions(model, tokenizer, text, device, args.n_steps)
            frac = fraction_on_identity(text, toks, attr)
            results.append({
                "id": int(row["ID"]),
                "text": text,
                "tokens": toks,
                "attributions": attr,
                "fraction_on_identity": frac,
                "true_label": int(row["Biased"]),
                "pred_prob": prob,
                "pred_label": int(prob >= 0.5),
            })
        except Exception as e:
            skipped += 1
            logger.warning(f"Skip example {i}: {type(e).__name__}: {e}")
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(test_df)} (skipped={skipped})")

    # Skip rate >2% usually means tokenizer/model-name mismatch; abort.
    skip_rate = skipped / max(1, len(test_df))
    if skip_rate > 0.02:
        raise RuntimeError(
            f"Skipped {skipped}/{len(test_df)} ({skip_rate:.1%}); aborts at >2%."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "tag": args.tag,
            "seed": args.seed,
            "n_examples": len(results),
            "examples": results,
        }, f)
    logger.info(f"Saved {len(results)} examples → {args.output}")


if __name__ == "__main__":
    main()
