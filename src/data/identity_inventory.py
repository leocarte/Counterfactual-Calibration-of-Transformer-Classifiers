"""Antisemitism-aware identity-token inventory for counterfactual generation.

Defines four identity categories (religious-ethnic, geopolitical proxy, slurs,
implicit/conspiracy) plus the alternative identity groups available for swaps.

This is the antisemitism-specific innovation over Davani et al. WOAH 2021,
which used a generic SGT inventory across religion/ethnicity. Here:

  * geopolitical-proxy terms (israel, zionist) are recognized as IHRA-defined
    proxies for Jewish identity in antisemitic discourse, and are NOT swapped
    indiscriminately to other religions (no clean counterpart);
  * slurs are flagged for asymmetry filtering (a swap from "kike" to "muslim"
    changes the label, since the slur category is identity-specific);
  * implicit/conspiracy markers (globalist, "they control the X") are tracked
    so future LLM-filtering can score their plausibility under swap.

References
----------
Mostafazadeh Davani et al., "Improving Counterfactual Generation for Fair Hate
Speech Detection," WOAH 2021. arXiv:2108.01721.
Garg et al., "Counterfactual Fairness in Text Classification through Robustness,"
AIES 2019. arXiv:1809.10610.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class IdentityCategory(Enum):
    """Four-way taxonomy of antisemitism-relevant identity tokens."""

    RELIGIOUS_ETHNIC = "religious_ethnic"
    GEOPOLITICAL_PROXY = "geopolitical_proxy"
    SLUR = "slur"
    IMPLICIT_CONSPIRACY = "implicit_conspiracy"


# Lower-case canonical tokens. Matching is case-insensitive and punctuation-stripped.
_RELIGIOUS_ETHNIC_JEWISH = ("jew", "jews", "jewish", "hebrew", "hebrews", "judaism")
_GEOPOLITICAL_PROXY_JEWISH = (
    "israel", "israeli", "israelis",
    "zionist", "zionists", "zionism",
)
_SLURS_JEWISH = (
    "kike", "kikes", "kyke",
    "zionazi", "zionazis",
    "christkiller", "christkillers",
)
_IMPLICIT_CONSPIRACY_JEWISH = (
    "globalist", "globalists",
    "rothschild", "rothschilds",
    "soros",
    # NB: phrase patterns such as "they control the media" require regex matching
    # and are handled by `find_implicit_conspiracy_phrases()` below, not this list.
)


# Alternative identity groups for swaps.
# Each entry is (group_label, [tokens that ground that group in the surface form]).
# We swap *only* religious-ethnic Jewish tokens to these alternatives — slurs,
# geopolitical proxies, and conspiracy markers are NOT freely swapped because
# they have no clean cross-religion counterpart.
ALTERNATIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "muslim":    ("muslim", "muslims", "islamic", "islam"),
    "christian": ("christian", "christians", "christianity"),
    "hindu":     ("hindu", "hindus", "hinduism"),
    "buddhist":  ("buddhist", "buddhists", "buddhism"),
    "atheist":   ("atheist", "atheists", "atheism"),
}

# For swap construction: each Jewish religious-ethnic token maps to a parallel
# token in each alternative group (singular noun, plural noun, adjective).
# Grammatical number/role is approximate — the LLM filter (Yasmin's vertical)
# will reject any swap that produces an ungrammatical sentence.
JEWISH_TO_ALTERNATIVE_TOKEN: dict[str, dict[str, str]] = {
    # singular noun / person
    "jew":    {"muslim": "muslim",    "christian": "christian", "hindu": "hindu",   "buddhist": "buddhist", "atheist": "atheist"},
    # plural noun / people
    "jews":   {"muslim": "muslims",   "christian": "christians", "hindu": "hindus", "buddhist": "buddhists", "atheist": "atheists"},
    # adjective form
    "jewish": {"muslim": "muslim",    "christian": "christian", "hindu": "hindu",   "buddhist": "buddhist", "atheist": "atheist"},
    # rarer ethnic term
    "hebrew": {"muslim": "muslim",    "christian": "christian", "hindu": "hindu",   "buddhist": "buddhist", "atheist": "atheist"},
    "hebrews": {"muslim": "muslims",  "christian": "christians", "hindu": "hindus", "buddhist": "buddhists", "atheist": "atheists"},
    "judaism": {"muslim": "islam",    "christian": "christianity", "hindu": "hinduism", "buddhist": "buddhism", "atheist": "atheism"},
}


@dataclass(frozen=True)
class IdentityMatch:
    """A single identity-token detection in a tweet.

    Attributes
    ----------
    token : str
        The matched surface form (preserves casing in the source text).
    canonical : str
        The lower-case, punctuation-stripped form used for inventory matching.
    category : IdentityCategory
        Which of the four taxonomy buckets this token falls in.
    start, end : int
        Character offsets into the original text (so we can splice swaps in).
    """

    token: str
    canonical: str
    category: IdentityCategory
    start: int
    end: int


# Pre-compiled regex for word-boundary matching. We need to scan tweets quickly;
# building the union once at import time is cheaper than per-call construction.
def _build_pattern() -> re.Pattern[str]:
    all_tokens: list[str] = []
    all_tokens.extend(_RELIGIOUS_ETHNIC_JEWISH)
    all_tokens.extend(_GEOPOLITICAL_PROXY_JEWISH)
    all_tokens.extend(_SLURS_JEWISH)
    all_tokens.extend(_IMPLICIT_CONSPIRACY_JEWISH)
    # Sort by length descending so multi-word tokens take precedence (none today,
    # but defensive for future additions).
    all_tokens.sort(key=len, reverse=True)
    escaped = [re.escape(t) for t in all_tokens]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


_TOKEN_PATTERN = _build_pattern()


def _categorize(canonical: str) -> IdentityCategory:
    if canonical in _RELIGIOUS_ETHNIC_JEWISH:
        return IdentityCategory.RELIGIOUS_ETHNIC
    if canonical in _GEOPOLITICAL_PROXY_JEWISH:
        return IdentityCategory.GEOPOLITICAL_PROXY
    if canonical in _SLURS_JEWISH:
        return IdentityCategory.SLUR
    if canonical in _IMPLICIT_CONSPIRACY_JEWISH:
        return IdentityCategory.IMPLICIT_CONSPIRACY
    raise ValueError(f"Unrecognized canonical token: {canonical!r}")


def find_identity_matches(text: str) -> list[IdentityMatch]:
    """Locate every Jewish-related identity-token occurrence in ``text``.

    Returns an empty list if none. Match order is left-to-right by start offset.
    Case-insensitive; punctuation handled by ``\\b`` word boundaries.
    """
    matches: list[IdentityMatch] = []
    for m in _TOKEN_PATTERN.finditer(text):
        token = m.group(1)
        canonical = token.lower()
        category = _categorize(canonical)
        matches.append(IdentityMatch(
            token=token,
            canonical=canonical,
            category=category,
            start=m.start(1),
            end=m.end(1),
        ))
    return matches


def has_jewish_identity_token(text: str) -> bool:
    """Cheap predicate: True iff any Jewish identity token is present.

    Used by the dataset to skip counterfactual generation on tweets where no
    swap is possible (i.e. CLP loss will be zero anyway).
    """
    return _TOKEN_PATTERN.search(text) is not None


def primary_keyword_group(text: str) -> str:
    """Assign one of {jews, israel, kikes, zionazi, none} to a tweet.

    Used by CCI v2 to look up the per-group temperature ``T_g``. Rule:
    first match wins, with priority (slur > proxy > religious-ethnic > none).
    Slurs go first because they are the strongest identity signal even when
    other categories are present.

    Note: ``kikes`` and ``zionazi`` are reported as separate groups because
    GoldStandard2024 keyword splits track them separately (per-keyword F1
    table, base rates 45.0 % and 91.1 % respectively).
    """
    matches = find_identity_matches(text)
    if not matches:
        return "none"

    # Slurs first
    for m in matches:
        if m.category is IdentityCategory.SLUR:
            if "kike" in m.canonical or "kyke" in m.canonical:
                return "kikes"
            if "zionazi" in m.canonical:
                return "zionazi"
            # christkiller etc. — fall through to next priority
    # Geopolitical proxies
    for m in matches:
        if m.category is IdentityCategory.GEOPOLITICAL_PROXY:
            return "israel"
    # Religious-ethnic
    for m in matches:
        if m.category is IdentityCategory.RELIGIOUS_ETHNIC:
            return "jews"
    # Implicit/conspiracy or unmatched: bucket as "jews"
    return "jews"


def get_swap_token(jewish_token: str, alternative_group: str) -> str | None:
    """Return the parallel token in ``alternative_group`` for a Jewish religious-
    ethnic ``jewish_token``, or None if no clean parallel exists.

    Returns None if:
      * the token is not in the religious-ethnic list (slurs/proxies are not freely
        swapped — that's the antisemitism-specific design choice);
      * the alternative group is unknown.
    """
    canonical = jewish_token.lower()
    if canonical not in JEWISH_TO_ALTERNATIVE_TOKEN:
        return None
    if alternative_group not in JEWISH_TO_ALTERNATIVE_TOKEN[canonical]:
        return None
    return JEWISH_TO_ALTERNATIVE_TOKEN[canonical][alternative_group]


# Phrase-level conspiracy markers that need regex (not single-token) matching.
# These are NOT used for swap generation — only for category-aware counting and
# future LLM-filter scoring. v0 ships with a small list; expand opportunistically.
_CONSPIRACY_PHRASE_PATTERNS = [
    re.compile(r"\bthey\s+control\s+(the\s+)?(media|banks?|world|government|hollywood)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+world\s+order\b", re.IGNORECASE),
    re.compile(r"\bglobal\s+(elite|cabal)\b", re.IGNORECASE),
]


def find_implicit_conspiracy_phrases(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, matched_text) for any conspiracy phrase patterns."""
    results: list[tuple[int, int, str]] = []
    for pattern in _CONSPIRACY_PHRASE_PATTERNS:
        for m in pattern.finditer(text):
            results.append((m.start(), m.end(), m.group(0)))
    return results


# Public, frozen exports for inspection in notebooks/tests.
RELIGIOUS_ETHNIC_TOKENS: frozenset[str] = frozenset(_RELIGIOUS_ETHNIC_JEWISH)
GEOPOLITICAL_PROXY_TOKENS: frozenset[str] = frozenset(_GEOPOLITICAL_PROXY_JEWISH)
SLUR_TOKENS: frozenset[str] = frozenset(_SLURS_JEWISH)
IMPLICIT_CONSPIRACY_TOKENS: frozenset[str] = frozenset(_IMPLICIT_CONSPIRACY_JEWISH)
ALL_JEWISH_IDENTITY_TOKENS: frozenset[str] = frozenset(
    RELIGIOUS_ETHNIC_TOKENS
    | GEOPOLITICAL_PROXY_TOKENS
    | SLUR_TOKENS
    | IMPLICIT_CONSPIRACY_TOKENS
)
ALTERNATIVE_GROUP_NAMES: tuple[str, ...] = tuple(ALTERNATIVE_GROUPS.keys())

# CCI v2 group label → integer index mapping. Used by TemperatureHead.
PRIMARY_KEYWORD_GROUPS: tuple[str, ...] = ("jews", "israel", "kikes", "zionazi", "none")
GROUP_TO_INDEX: dict[str, int] = {g: i for i, g in enumerate(PRIMARY_KEYWORD_GROUPS)}


def group_label_to_index(label: str) -> int:
    """Convert a primary-keyword-group label to a 0-indexed integer.

    Raises ValueError on unknown labels (callers should use ``primary_keyword_group``
    which only emits known labels).
    """
    if label not in GROUP_TO_INDEX:
        raise ValueError(
            f"Unknown group label {label!r}; expected one of {PRIMARY_KEYWORD_GROUPS}"
        )
    return GROUP_TO_INDEX[label]
