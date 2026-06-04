"""Robustness and fairness evaluation."""
import re

import numpy as np
import pandas as pd
from typing import Callable

from src.data.keyword_masking import keyword_sensitivity_test
from src.evaluation.metrics import compute_metrics


def temporal_robustness(
    y_true: list[int],
    y_pred: list[int],
    dates: list[str],
    cutoff_year: int = 2022,
) -> dict:
    """
    Compare model performance on pre-cutoff vs. post-cutoff tweets.

    Tests whether the model generalizes across time periods
    (discourse evolves, new coded language emerges).
    """
    dates_parsed = pd.to_datetime(dates, errors="coerce")

    pre_mask = dates_parsed.year < cutoff_year
    post_mask = dates_parsed.year >= cutoff_year

    results = {}

    for name, mask in [("pre_2022", pre_mask), ("2022_onwards", post_mask)]:
        yt = [y for y, m in zip(y_true, mask) if m]
        yp = [y for y, m in zip(y_pred, mask) if m]

        if len(yt) > 0:
            from sklearn.metrics import f1_score
            results[name] = {
                "n": len(yt),
                "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
            }

    return results


def identity_term_bias_test(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    identity_terms: list[str] = None,
) -> dict:
    """
    Measure false positive rate on tweets containing identity terms
    in non-antisemitic contexts.

    A high FPR on benign identity mentions indicates the model is
    biased by the presence of the term rather than understanding context.
    """
    if identity_terms is None:
        identity_terms = ["jewish", "jews", "israel", "israeli"]

    # Find non-antisemitic tweets containing identity terms (word-boundary match
    # to avoid false positives like "israel" matching "israelite")
    term_patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in identity_terms]
    benign_with_identity = []
    for i, (text, label) in enumerate(zip(texts, y_true)):
        if label == 0 and any(p.search(text) for p in term_patterns):
            benign_with_identity.append(i)

    if not benign_with_identity:
        return {"fpr_on_identity_mentions": 0.0, "n_benign_with_identity": 0}

    # FPR = false positives among benign identity-containing tweets
    false_positives = sum(1 for i in benign_with_identity if y_pred[i] == 1)
    fpr = false_positives / len(benign_with_identity)

    return {
        "fpr_on_identity_mentions": fpr,
        "n_benign_with_identity": len(benign_with_identity),
        "n_false_positives": false_positives,
    }
