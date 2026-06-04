"""Tests for src/data/symmetry_classifier.py."""
import pytest

from src.data.symmetry_classifier import (
    HeuristicSymmetryClassifier,
    LLMSymmetryClassifier,
    SymmetryDecision,
)


class TestHeuristicSymmetryClassifier:
    def setup_method(self):
        self.clf = HeuristicSymmetryClassifier()

    def test_rule_1_slur_rejected(self):
        # Source contains a slur — any swap is asymmetric
        decision = self.clf.decide(
            "kike behavior",
            "muslim behavior",
            ("kike", "muslim"),
        )
        assert decision.is_symmetric is False
        assert "rule_1" in decision.reason

    def test_rule_2_proxy_only_rejected(self):
        # Source contains only proxy ("israel"), no religious-ethnic anchor
        decision = self.clf.decide(
            "Israel does X",
            "Egypt does X",
            ("israel", "egypt"),
        )
        assert decision.is_symmetric is False
        assert "rule_2" in decision.reason

    def test_rule_3_religious_only_accepted(self):
        decision = self.clf.decide(
            "I respect Jewish people",
            "I respect Muslim people",
            ("Jewish", "Muslim"),
        )
        assert decision.is_symmetric is True
        assert "rule_3" in decision.reason

    def test_rule_3_religious_with_proxy_accepted(self):
        # Religious-ethnic + proxy together: swap touches the religious-ethnic
        # token, the proxy survives unchanged → symmetric
        decision = self.clf.decide(
            "Jewish settlers in Israel",
            "Muslim settlers in Israel",
            ("Jewish", "Muslim"),
        )
        assert decision.is_symmetric is True

    def test_rule_4_no_identity_token_rejected(self):
        decision = self.clf.decide("hello world", "hello world", ("a", "b"))
        assert decision.is_symmetric is False
        assert "rule_0" in decision.reason  # no identity token at all

    def test_implicit_only_rejected(self):
        # Implicit conspiracy term but no religious-ethnic anchor → conservative
        decision = self.clf.decide(
            "the globalist agenda",
            "the globalist agenda",
            ("globalist", "globalist"),
        )
        assert decision.is_symmetric is False
        assert "rule_4" in decision.reason


class _MockLLMBackend:
    """Test double for LLMSymmetryClassifier."""

    def __init__(self, log_p_x: float, log_p_swap: float, label_preserved: bool):
        self.log_p_x = log_p_x
        self.log_p_swap = log_p_swap
        self.label_preserved = label_preserved

    def score_log_likelihood(self, text: str) -> float:
        # Crude switch — first call returns x, second returns swap.
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        return self.log_p_x if self._call_count == 1 else self.log_p_swap

    def judges_label_preserved(self, x: str, x_swap: str) -> bool:
        return self.label_preserved


class TestLLMSymmetryClassifier:
    def test_no_backend_falls_back_to_heuristic(self):
        clf = LLMSymmetryClassifier(llm_backend=None)
        decision = clf.decide(
            "I respect Jewish people",
            "I respect Muslim people",
            ("Jewish", "Muslim"),
        )
        assert decision.is_symmetric is True
        assert "fallback" in decision.reason

    def test_likelihood_too_different_rejected(self):
        backend = _MockLLMBackend(log_p_x=-2.0, log_p_swap=-10.0, label_preserved=True)
        clf = LLMSymmetryClassifier(llm_backend=backend, log_likelihood_threshold=1.5)
        decision = clf.decide("a", "b", ("a", "b"))
        assert decision.is_symmetric is False
        assert "implausible" in decision.reason

    def test_label_flip_rejected(self):
        backend = _MockLLMBackend(log_p_x=-2.0, log_p_swap=-2.5, label_preserved=False)
        clf = LLMSymmetryClassifier(llm_backend=backend, log_likelihood_threshold=1.5)
        decision = clf.decide("a", "b", ("a", "b"))
        assert decision.is_symmetric is False
        assert "label_flip" in decision.reason

    def test_likelihood_and_label_pass(self):
        backend = _MockLLMBackend(log_p_x=-2.0, log_p_swap=-2.5, label_preserved=True)
        clf = LLMSymmetryClassifier(llm_backend=backend, log_likelihood_threshold=1.5)
        decision = clf.decide("a", "b", ("a", "b"))
        assert decision.is_symmetric is True
        assert "pass" in decision.reason


class TestSymmetryDecision:
    def test_immutable(self):
        d = SymmetryDecision(True, "test")
        with pytest.raises(Exception):
            d.is_symmetric = False  # frozen dataclass
