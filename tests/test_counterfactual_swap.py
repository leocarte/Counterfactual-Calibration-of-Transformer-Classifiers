"""Tests for src/data/counterfactual_swap.py."""
import random

import pytest

from src.data.counterfactual_swap import (
    CounterfactualPair,
    CounterfactualSwapEngine,
    compute_swap_yield,
)
from src.data.symmetry_classifier import HeuristicSymmetryClassifier


@pytest.fixture
def deterministic_engine():
    """An engine with a fixed RNG seed for reproducible swap sampling."""
    return CounterfactualSwapEngine(rng=random.Random(42))


class TestCounterfactualSwapEngine:
    def test_no_identity_token_returns_empty(self, deterministic_engine):
        assert deterministic_engine.generate("hello world") == []

    def test_no_eligible_token_returns_empty(self, deterministic_engine):
        # "israel" is a proxy, "kike" is a slur — neither swappable
        assert deterministic_engine.generate("Israel is a country") == []
        assert deterministic_engine.generate("kike is a slur") == []

    def test_religious_ethnic_yields_swaps(self, deterministic_engine):
        pairs = deterministic_engine.generate("I think Jewish people are great")
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, CounterfactualPair)
            assert p.original_token == "Jewish"
            assert p.swap_token in {"Muslim", "Christian", "Hindu", "Buddhist", "Atheist"}
            assert "Jewish" not in p.swap_text
            assert p.swap_token in p.swap_text

    def test_swap_preserves_casing(self, deterministic_engine):
        pairs_lower = deterministic_engine.generate("jewish people are diverse")
        pairs_upper = deterministic_engine.generate("JEWISH people are diverse")
        for p in pairs_lower:
            assert p.swap_token.islower()
        for p in pairs_upper:
            assert p.swap_token.isupper()

    def test_swap_preserves_position(self, deterministic_engine):
        text = "I respect Jewish people"
        pairs = deterministic_engine.generate(text)
        for p in pairs:
            start, end = p.swap_position
            assert text[start:end] == "Jewish"
            # Reconstruct: text with that span replaced by swap_token == p.swap_text
            assert text[:start] + p.swap_token + text[end:] == p.swap_text

    def test_pairs_cap_respected(self):
        engine = CounterfactualSwapEngine(
            swaps_per_token=5,
            pairs_cap_per_example=2,
            rng=random.Random(42),
        )
        # Multi-token text could yield up to 5 + 5 = 10 pairs
        pairs = engine.generate("Jewish people and Jewish leaders are diverse")
        assert len(pairs) <= 2

    def test_swaps_per_token_respected(self):
        engine = CounterfactualSwapEngine(
            swaps_per_token=2,
            pairs_cap_per_example=10,
            rng=random.Random(42),
        )
        pairs = engine.generate("Jewish people are diverse")
        # At most 2 swaps from one token
        assert len(pairs) <= 2

    def test_seeded_engine_is_reproducible(self):
        e1 = CounterfactualSwapEngine(rng=random.Random(42))
        e2 = CounterfactualSwapEngine(rng=random.Random(42))
        text = "Jewish people are diverse"
        pairs_1 = e1.generate(text)
        pairs_2 = e2.generate(text)
        assert [p.swap_token for p in pairs_1] == [p.swap_token for p in pairs_2]

    def test_invalid_swaps_per_token_raises(self):
        with pytest.raises(ValueError):
            CounterfactualSwapEngine(swaps_per_token=0)
        with pytest.raises(ValueError):
            CounterfactualSwapEngine(swaps_per_token=99)

    def test_invalid_pairs_cap_raises(self):
        with pytest.raises(ValueError):
            CounterfactualSwapEngine(pairs_cap_per_example=0)

    def test_slur_in_text_yields_no_pairs(self, deterministic_engine):
        # Text has "Jewish" (eligible) + "kike" (slur). Symmetry filter rejects.
        pairs = deterministic_engine.generate("Jewish people are not kikes")
        assert pairs == []

    def test_proxy_only_yields_no_pairs(self, deterministic_engine):
        # Text has only proxy — religious-ethnic absent — no eligible match
        pairs = deterministic_engine.generate("Israel is in the news today")
        assert pairs == []


class TestComputeSwapYield:
    def test_basic_stats(self):
        engine = CounterfactualSwapEngine(rng=random.Random(42))
        texts = [
            "hello world",  # no identity
            "jewish people are great",  # eligible
            "Israel is in the news",  # identity present, no eligible swap
            "another jewish leader speaks",  # eligible
        ]
        stats = compute_swap_yield(texts, engine)
        assert stats["total_texts"] == 4
        assert stats["texts_with_identity"] == 3  # 3 contain at least one identity token
        assert stats["texts_with_pairs"] == 2  # 2 yield symmetric swaps
        assert stats["mean_pairs_per_identity_text"] > 0
