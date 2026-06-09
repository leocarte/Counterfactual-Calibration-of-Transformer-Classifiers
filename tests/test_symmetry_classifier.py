"""Tests for src/data/symmetry_classifier.py."""
import pytest

from src.data.symmetry_classifier import (
    HeuristicSymmetryClassifier,
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


class TestSymmetryDecision:
    def test_immutable(self):
        d = SymmetryDecision(True, "test")
        with pytest.raises(Exception):
            d.is_symmetric = False  # frozen dataclass
