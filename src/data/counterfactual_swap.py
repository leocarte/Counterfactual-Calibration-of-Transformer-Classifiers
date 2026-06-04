"""Counterfactual swap engine — generates Φ(x) for the CLP / CCI v2 losses.

For every Jewish religious-ethnic token in a tweet, samples K alternative-group
substitutions and (optionally) filters through a :class:`SymmetryClassifier` so
only label-preserving swaps survive. Caps total pairs per example to keep batch
shapes manageable.

This module is loss-agnostic: both the Davani-style CLP loss (logit-space) and
the CCI v2 loss (probability-space, per-group temperature) consume the same
``CounterfactualPair`` output.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from src.data.identity_inventory import (
    ALTERNATIVE_GROUP_NAMES,
    IdentityCategory,
    IdentityMatch,
    find_identity_matches,
    get_swap_token,
)
from src.data.symmetry_classifier import (
    HeuristicSymmetryClassifier,
    SymmetryClassifier,
    SymmetryDecision,
)


@dataclass(frozen=True)
class CounterfactualPair:
    """A single (x → x_swap) substitution that passed symmetry filtering.

    Attributes
    ----------
    original_text : str
        Source tweet, unchanged from the input.
    swap_text : str
        Counterfactual: same tweet with one Jewish religious-ethnic token
        substituted for an alternative-group equivalent.
    original_token : str
        The Jewish token that was substituted (preserves source casing).
    swap_token : str
        The alternative-group token spliced in (matches source casing).
    swap_position : tuple[int, int]
        (start, end) char offsets of the substitution in ``original_text``.
    alternative_group : str
        Which alternative group the swap targeted ("muslim", "christian", ...).
    decision : SymmetryDecision
        Why the symmetry classifier accepted this pair (auditable).
    """

    original_text: str
    swap_text: str
    original_token: str
    swap_token: str
    swap_position: tuple[int, int]
    alternative_group: str
    decision: SymmetryDecision


def _match_casing(target: str, source: str) -> str:
    """Return ``target`` with casing matched to ``source``.

    Heuristic:
      * if source is all upper → upper
      * if source is title-case (first letter upper) → title-case target
      * else → lowercase

    Designed to keep swaps tweet-faithful: ``Jewish people`` should become
    ``Muslim people``, not ``muslim people``.
    """
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target.lower()


def _splice(text: str, start: int, end: int, replacement: str) -> str:
    """Return ``text`` with the ``[start:end)`` slice replaced by ``replacement``."""
    return text[:start] + replacement + text[end:]


class CounterfactualSwapEngine:
    """Generate symmetry-filtered counterfactual swaps for a tweet.

    Parameters
    ----------
    classifier : SymmetryClassifier, optional
        Decides which swaps are label-preserving. Defaults to
        :class:`HeuristicSymmetryClassifier` (no LLM dependency).
    swaps_per_token : int, default 3
        Number of alternative groups to sample per Jewish identity token.
        Sampled without replacement from the 5 available alternatives.
    pairs_cap_per_example : int, default 5
        Hard cap on total counterfactual pairs returned per tweet. Prevents
        identity-rich tweets from dominating the CLP gradient.
    rng : random.Random, optional
        Pre-seeded RNG for reproducible swap sampling. If None, uses the
        module-level ``random`` (seed via ``random.seed(...)`` before training).
    """

    def __init__(
        self,
        classifier: Optional[SymmetryClassifier] = None,
        swaps_per_token: int = 3,
        pairs_cap_per_example: int = 5,
        rng: Optional[random.Random] = None,
    ):
        if not 1 <= swaps_per_token <= len(ALTERNATIVE_GROUP_NAMES):
            raise ValueError(
                f"swaps_per_token must be in [1, {len(ALTERNATIVE_GROUP_NAMES)}]; "
                f"got {swaps_per_token}"
            )
        if pairs_cap_per_example < 1:
            raise ValueError(f"pairs_cap_per_example must be ≥ 1; got {pairs_cap_per_example}")
        self.classifier = classifier or HeuristicSymmetryClassifier()
        self.swaps_per_token = swaps_per_token
        self.pairs_cap_per_example = pairs_cap_per_example
        self.rng = rng or random

    def generate(self, text: str) -> list[CounterfactualPair]:
        """Generate up to ``pairs_cap_per_example`` symmetry-filtered swaps.

        Returns an empty list if no Jewish religious-ethnic tokens are present
        or all candidate swaps fail the symmetry filter.
        """
        matches = find_identity_matches(text)
        # Only religious-ethnic tokens are eligible for swapping. Slurs and
        # geopolitical proxies have no clean cross-religion counterpart by
        # design (see identity_inventory.py).
        eligible = [m for m in matches if m.category is IdentityCategory.RELIGIOUS_ETHNIC]
        if not eligible:
            return []

        pairs: list[CounterfactualPair] = []
        for match in eligible:
            if len(pairs) >= self.pairs_cap_per_example:
                break
            pairs.extend(self._swaps_for_match(text, match))
            pairs = pairs[: self.pairs_cap_per_example]
        return pairs

    def _swaps_for_match(
        self,
        text: str,
        match: IdentityMatch,
    ) -> list[CounterfactualPair]:
        """All accepted swap pairs for a single token occurrence."""
        # Sample K alternative groups without replacement.
        groups = list(ALTERNATIVE_GROUP_NAMES)
        self.rng.shuffle(groups)
        sampled_groups = groups[: self.swaps_per_token]

        accepted: list[CounterfactualPair] = []
        for group in sampled_groups:
            swap_token_canonical = get_swap_token(match.canonical, group)
            if swap_token_canonical is None:
                continue
            # Match source casing
            swap_token = _match_casing(swap_token_canonical, match.token)
            swap_text = _splice(text, match.start, match.end, swap_token)
            decision = self.classifier.decide(text, swap_text, (match.token, swap_token))
            if decision.is_symmetric:
                accepted.append(CounterfactualPair(
                    original_text=text,
                    swap_text=swap_text,
                    original_token=match.token,
                    swap_token=swap_token,
                    swap_position=(match.start, match.end),
                    alternative_group=group,
                    decision=decision,
                ))
        return accepted


def compute_swap_yield(
    texts: list[str],
    engine: CounterfactualSwapEngine,
) -> dict:
    """Diagnostic: report per-corpus pair-yield statistics.

    Used to verify that the heuristic filter is not too aggressive (target:
    ≥1 pair on average for tweets that contain a Jewish identity token).

    Returns
    -------
    dict with keys:
        ``total_texts``, ``texts_with_identity``, ``texts_with_pairs``,
        ``mean_pairs_per_identity_text``, ``rejected_by_reason`` (counter).
    """
    from collections import Counter

    total = len(texts)
    n_with_identity = 0
    n_with_pairs = 0
    pair_counts: list[int] = []
    rejection_reasons: Counter = Counter()

    for text in texts:
        matches = find_identity_matches(text)
        if not matches:
            continue
        n_with_identity += 1
        pairs = engine.generate(text)
        if pairs:
            n_with_pairs += 1
            pair_counts.append(len(pairs))

    mean = sum(pair_counts) / max(1, len(pair_counts))
    return {
        "total_texts": total,
        "texts_with_identity": n_with_identity,
        "texts_with_pairs": n_with_pairs,
        "mean_pairs_per_identity_text": mean,
        # rejection_reasons would require the engine to expose its filter log;
        # left as a TODO for the LLM-filter integration where reason-counting matters.
        "rejected_by_reason": dict(rejection_reasons),
    }
