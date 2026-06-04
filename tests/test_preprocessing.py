"""Tests for preprocessing pipeline."""
import pytest
from src.data.preprocessing import preprocess_tweet, mask_keywords


class TestPreprocessTweet:
    def test_username_masking(self):
        result = preprocess_tweet("@JohnDoe said something about @JaneSmith")
        assert "@USER" in result
        assert "@JohnDoe" not in result

    def test_url_removal(self):
        result = preprocess_tweet("Check this https://example.com/page link")
        assert "[URL]" in result
        assert "https://" not in result

    def test_hashtag_preserved(self):
        result = preprocess_tweet("#FreePalestine is trending")
        assert "FreePalestine" in result
        assert "#" not in result

    def test_empty_string(self):
        assert preprocess_tweet("") == ""

    def test_non_string(self):
        assert preprocess_tweet(None) == ""

    def test_casing_preserved(self):
        result = preprocess_tweet("THIS IS IMPORTANT", lowercase=False)
        assert result == "THIS IS IMPORTANT"

    def test_casing_lowered(self):
        result = preprocess_tweet("THIS IS IMPORTANT", lowercase=True)
        assert result == "this is important"


class TestMaskKeywords:
    def test_basic_masking(self):
        result = mask_keywords("The Jews are celebrating Hanukkah")
        assert "[MASK]" in result
        assert "celebrating Hanukkah" in result

    def test_preserves_punctuation(self):
        result = mask_keywords("Jews, Israel, and more")
        assert "[MASK]," in result or "[MASK]" in result

    def test_case_insensitive(self):
        result = mask_keywords("JEWS and jews and Jews")
        assert result.count("[MASK]") == 3

    def test_no_keywords(self):
        result = mask_keywords("This tweet has no relevant terms")
        assert "[MASK]" not in result
