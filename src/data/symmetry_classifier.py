"""Counterfactual symmetry classifier — decides whether a swap preserves the label.

Davani et al. WOAH 2021's central insight: enforcing ``f(x) = f(x_swap)`` makes
sense only when ``y(x) = y(x_swap)``. For asymmetric swaps (e.g. swapping a slur
out, or swapping a geopolitical proxy across religions where no parallel exists),
the swap implicitly changes the label and the CLP penalty destroys real signal.

Two implementations are provided:

  * :class:`HeuristicSymmetryClassifier` — token-rule based, no model dependency.
    Ships immediately, used for Phase 1 Davani runs.
  * :class:`LLMSymmetryClassifier` — language-model likelihood scoring of
    ``P(x_swap | swap_template)`` vs ``P(x | swap_template)``. Plumbed in by
    Yasmin's vertical once the LLM backend is online.

Both expose the same :class:`SymmetryClassifier` interface so the CLP pipeline
is LLM-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.data.identity_inventory import (
    IdentityCategory,
    find_identity_matches,
)


@dataclass(frozen=True)
class SymmetryDecision:
    """Outcome of classifying a single (x, x_swap) counterfactual pair."""

    is_symmetric: bool
    reason: str  # short string for logging / audit (e.g. "rule_3_religious_only")


class SymmetryClassifier(ABC):
    """Abstract interface — all symmetry classifiers must implement ``decide``."""

    @abstractmethod
    def decide(
        self,
        x: str,
        x_swap: str,
        swap_pair: tuple[str, str],
    ) -> SymmetryDecision:
        """Return whether the swap preserves the antisemitism label.

        Parameters
        ----------
        x : str
            Original tweet.
        x_swap : str
            Counterfactual tweet (Jewish identity token replaced).
        swap_pair : tuple[str, str]
            (original_token, swap_token) actually substituted.
        """
        ...

    def is_symmetric(self, x: str, x_swap: str, swap_pair: tuple[str, str]) -> bool:
        """Convenience boolean accessor."""
        return self.decide(x, x_swap, swap_pair).is_symmetric


class HeuristicSymmetryClassifier(SymmetryClassifier):
    """Token-rule symmetry classifier — Phase 1 default, no LLM dependency.

    Rules (applied in order):
      1. If ``x`` contains a slur (kike, zionazi, ...) → ASYMMETRIC.
         Slurs are identity-specific; swapping the slur out fundamentally
         changes the meaning even when the surrounding text is preserved.
      2. If ``x`` contains a geopolitical-proxy token (israel, zionist) and the
         surrounding context lacks a religion-symmetric counterpart → ASYMMETRIC.
         Conservative: if both x and x_swap retain the proxy token, treat as
         symmetric (e.g. "Israel and Saudi Arabia disagree" swapping
         Jewish→Muslim doesn't touch the proxy and is fine).
         More aggressive: if the proxy token is the ONLY identity reference,
         we cannot symmetrically swap to e.g. "Egypt" — reject.
      3. If ``x`` contains a religious-ethnic Jewish token (jew, jewish) and
         no slur or proxy is present → SYMMETRIC. The swap to muslim/christian/
         hindu/buddhist/atheist is meaning-preserving in this regime.
      4. Conservative default (uncertain) → ASYMMETRIC.

    Acceptance rate on a typical tweet stream: ~40-60 %, deliberately
    conservative. The CLP penalty applied to fewer but cleaner pairs is
    expected to outperform the same penalty on many noisier pairs (see
    Davani Table 2).
    """

    def decide(
        self,
        x: str,
        x_swap: str,
        swap_pair: tuple[str, str],
    ) -> SymmetryDecision:
        matches = find_identity_matches(x)
        if not matches:
            return SymmetryDecision(False, "rule_0_no_identity_token_in_x")

        has_slur = any(m.category is IdentityCategory.SLUR for m in matches)
        has_proxy = any(m.category is IdentityCategory.GEOPOLITICAL_PROXY for m in matches)
        has_religious = any(
            m.category is IdentityCategory.RELIGIOUS_ETHNIC for m in matches
        )

        # Rule 1: slur → asymmetric
        if has_slur:
            return SymmetryDecision(False, "rule_1_slur_present")

        # Rule 2: geopolitical proxy as sole identity → asymmetric
        if has_proxy and not has_religious:
            return SymmetryDecision(
                False, "rule_2_proxy_only_no_religion_symmetric_swap"
            )

        # Rule 3: religious-ethnic only (or religious + proxy that survives) →
        # symmetric. The swap acts on the religious-ethnic token; the proxy
        # term, if any, is unchanged in x_swap.
        if has_religious:
            return SymmetryDecision(True, "rule_3_religious_only_or_with_proxy")

        # Rule 4: conservative fallthrough — implicit/conspiracy markers
        # without religious-ethnic token are rare and ambiguous. Reject.
        return SymmetryDecision(False, "rule_4_conservative_default")


class LLMSymmetryClassifier(SymmetryClassifier):
    """Symmetry classifier driven by language-model likelihood scoring.

    Plumbed in by Yasmin's vertical once the LLM backend (Mistral-7B-Instruct
    or, post-gate, Llama-3.1-8B-Instruct) is reachable from the training pod.

    The decision criterion is two-step:

      1. **Likelihood plausibility** — compute ``log P(x_swap)`` and ``log P(x)``
         under the LLM. If ``|log P(x) - log P(x_swap)| > threshold``, the swap
         produces an ungrammatical / implausible sentence (e.g. an "Israel"
         token replaced by "Egypt" in a religious context). Reject.
      2. **Label-preserving judgment** — prompt the LLM with the original and
         swapped tweets and ask whether the antisemitism label should remain
         unchanged. Reject if the LLM signals a label flip.

    v1 ships as a stub that defers to a heuristic fallback whenever the LLM
    backend is not configured. This keeps Phase 1 unblocked.
    """

    def __init__(
        self,
        llm_backend: Optional[object] = None,
        log_likelihood_threshold: float = 1.5,
        fallback: Optional[SymmetryClassifier] = None,
    ):
        self.llm_backend = llm_backend
        self.log_likelihood_threshold = log_likelihood_threshold
        self.fallback = fallback or HeuristicSymmetryClassifier()

    def decide(
        self,
        x: str,
        x_swap: str,
        swap_pair: tuple[str, str],
    ) -> SymmetryDecision:
        if self.llm_backend is None:
            decision = self.fallback.decide(x, x_swap, swap_pair)
            return SymmetryDecision(
                decision.is_symmetric,
                f"llm_backend_unset_fallback_{decision.reason}",
            )

        # Step 1: likelihood plausibility — caller-injected backend must implement
        # ``score_log_likelihood(text: str) -> float``.
        log_p_x = self.llm_backend.score_log_likelihood(x)
        log_p_swap = self.llm_backend.score_log_likelihood(x_swap)
        if abs(log_p_x - log_p_swap) > self.log_likelihood_threshold:
            return SymmetryDecision(
                False,
                f"llm_implausible_swap_dll={log_p_x - log_p_swap:.2f}",
            )

        # Step 2: label-preserving judgment — caller-injected backend must
        # implement ``judges_label_preserved(x: str, x_swap: str) -> bool``.
        if not self.llm_backend.judges_label_preserved(x, x_swap):
            return SymmetryDecision(False, "llm_label_flip_judged")

        return SymmetryDecision(True, "llm_likelihood_and_label_pass")
