"""Keyword masking utilities for counterfactual augmentation."""
import re
import random
from typing import Optional

from src.data.preprocessing import ANTISEMITISM_KEYWORDS


def keyword_sensitivity_test(
    texts: list[str],
    predictions_original: list[int],
    predict_fn,
    keywords: Optional[list[str]] = None,
) -> dict:
    """
    Test how sensitive the model is to keyword presence.

    For each text, mask keywords and re-run prediction.
    Report the flip rate (fraction of predictions that change).

    A high flip rate (>30%) indicates keyword dependency rather than
    contextual understanding.
    """
    if keywords is None:
        keywords = ANTISEMITISM_KEYWORDS

    keywords_set = set(keywords)

    masked_texts = []
    has_keyword = []

    for text in texts:
        tokens = text.split()
        contains_kw = False
        masked_tokens = []
        for token in tokens:
            clean = re.sub(r"[^\w]", "", token).lower()
            if clean in keywords_set:
                contains_kw = True
                masked_tokens.append("[MASK]")
            else:
                masked_tokens.append(token)
        masked_texts.append(" ".join(masked_tokens))
        has_keyword.append(contains_kw)

    # Get predictions on masked texts
    predictions_masked = predict_fn(masked_texts)

    # Compute flip rate (only on texts that contain keywords)
    flips = 0
    total_with_keywords = 0

    for orig, masked, has_kw in zip(predictions_original, predictions_masked, has_keyword):
        if has_kw:
            total_with_keywords += 1
            if orig != masked:
                flips += 1

    flip_rate = flips / total_with_keywords if total_with_keywords > 0 else 0.0

    return {
        "flip_rate": flip_rate,
        "total_with_keywords": total_with_keywords,
        "total_flips": flips,
        "interpretation": (
            "HIGH keyword dependency — model uses shortcuts"
            if flip_rate > 0.3
            else "Moderate keyword sensitivity"
            if flip_rate > 0.15
            else "LOW keyword dependency — model uses context"
        ),
    }
