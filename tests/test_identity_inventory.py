"""Tests for src/data/identity_inventory.py."""
import pytest

from src.data.identity_inventory import (
    ALTERNATIVE_GROUP_NAMES,
    GROUP_TO_INDEX,
    PRIMARY_KEYWORD_GROUPS,
    IdentityCategory,
    find_identity_matches,
    find_implicit_conspiracy_phrases,
    get_swap_token,
    group_label_to_index,
    has_jewish_identity_token,
    primary_keyword_group,
)


class TestFindIdentityMatches:
    def test_finds_religious_ethnic(self):
        matches = find_identity_matches("I think Jewish people are great")
        assert len(matches) == 1
        assert matches[0].canonical == "jewish"
        assert matches[0].category is IdentityCategory.RELIGIOUS_ETHNIC

    def test_finds_geopolitical_proxy(self):
        matches = find_identity_matches("Israel and Palestine remain in conflict")
        assert any(m.canonical == "israel" for m in matches)
        assert all(m.category is IdentityCategory.GEOPOLITICAL_PROXY
                   for m in matches if m.canonical == "israel")

    def test_finds_slur(self):
        matches = find_identity_matches("Stop being a kike about it")
        slurs = [m for m in matches if m.category is IdentityCategory.SLUR]
        assert len(slurs) == 1
        assert slurs[0].canonical == "kike"

    def test_finds_implicit_conspiracy(self):
        matches = find_identity_matches("The globalist agenda is everywhere")
        implicit = [m for m in matches
                    if m.category is IdentityCategory.IMPLICIT_CONSPIRACY]
        assert len(implicit) == 1
        assert implicit[0].canonical == "globalist"

    def test_case_insensitive(self):
        for variant in ["JEWISH", "Jewish", "jewish", "JeWiSh"]:
            matches = find_identity_matches(f"{variant} community")
            assert len(matches) == 1
            assert matches[0].canonical == "jewish"
            # Source casing preserved in .token
            assert matches[0].token == variant

    def test_word_boundary(self):
        # "kikes" must match, but words containing the substring must not.
        # We don't currently have a real false-positive case, so test that
        # apostrophe-S-like patterns work and that punctuation doesn't kill
        # the match.
        matches = find_identity_matches("kikes, jews, and other people")
        canonicals = {m.canonical for m in matches}
        assert "kikes" in canonicals
        assert "jews" in canonicals

    def test_no_matches(self):
        assert find_identity_matches("the cat sat on the mat") == []

    def test_offsets_correct(self):
        text = "well, jewish people are diverse"
        matches = find_identity_matches(text)
        assert len(matches) == 1
        m = matches[0]
        assert text[m.start:m.end] == m.token

    def test_multiple_categories_in_one_text(self):
        text = "Jewish israelis kike globalist"
        matches = find_identity_matches(text)
        cats = [m.category for m in matches]
        assert IdentityCategory.RELIGIOUS_ETHNIC in cats
        assert IdentityCategory.GEOPOLITICAL_PROXY in cats
        assert IdentityCategory.SLUR in cats
        assert IdentityCategory.IMPLICIT_CONSPIRACY in cats


class TestHasJewishIdentityToken:
    def test_true_cases(self):
        assert has_jewish_identity_token("jewish people")
        assert has_jewish_identity_token("Israel is a country")
        assert has_jewish_identity_token("kike kike kike")
        assert has_jewish_identity_token("the Soros conspiracy")

    def test_false_cases(self):
        assert not has_jewish_identity_token("hello world")
        assert not has_jewish_identity_token("muslim people")
        assert not has_jewish_identity_token("")


class TestPrimaryKeywordGroup:
    def test_no_identity_returns_none(self):
        assert primary_keyword_group("hello world") == "none"

    def test_slur_kike_priority(self):
        # Slur should win over religious-ethnic + geopolitical
        assert primary_keyword_group("kike jewish israel") == "kikes"

    def test_slur_zionazi_priority(self):
        assert primary_keyword_group("zionazi jew jewish") == "zionazi"

    def test_proxy_when_no_slur(self):
        assert primary_keyword_group("jewish israel people") == "israel"

    def test_religious_ethnic(self):
        assert primary_keyword_group("jewish community") == "jews"

    def test_implicit_only_falls_to_jews(self):
        assert primary_keyword_group("the globalist agenda") == "jews"

    def test_all_known_groups_in_index(self):
        for g in PRIMARY_KEYWORD_GROUPS:
            assert g in GROUP_TO_INDEX
        assert len(GROUP_TO_INDEX) == len(PRIMARY_KEYWORD_GROUPS)


class TestGetSwapToken:
    def test_jewish_to_muslim(self):
        assert get_swap_token("jewish", "muslim") == "muslim"
        assert get_swap_token("jews", "muslim") == "muslims"

    def test_jewish_to_christian(self):
        assert get_swap_token("jewish", "christian") == "christian"

    def test_judaism_to_islam(self):
        # The religion-noun maps differently from the people-noun
        assert get_swap_token("judaism", "muslim") == "islam"
        assert get_swap_token("judaism", "buddhist") == "buddhism"

    def test_unknown_jewish_token_returns_none(self):
        # "kike" is a slur, not in the religious-ethnic mapping
        assert get_swap_token("kike", "muslim") is None

    def test_proxy_token_returns_none(self):
        # "israel" is a proxy, no clean cross-religion swap
        assert get_swap_token("israel", "muslim") is None

    def test_unknown_group_returns_none(self):
        assert get_swap_token("jewish", "klingon") is None


class TestGroupLabelToIndex:
    def test_known_labels(self):
        assert group_label_to_index("jews") == GROUP_TO_INDEX["jews"]
        assert group_label_to_index("none") == GROUP_TO_INDEX["none"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            group_label_to_index("not_a_group")


class TestFindImplicitConspiracyPhrases:
    def test_they_control_pattern(self):
        results = find_implicit_conspiracy_phrases("they control the media")
        assert len(results) == 1
        assert "control" in results[0][2].lower()

    def test_new_world_order(self):
        results = find_implicit_conspiracy_phrases("the New World Order is real")
        assert len(results) == 1

    def test_no_match(self):
        assert find_implicit_conspiracy_phrases("hello world") == []
