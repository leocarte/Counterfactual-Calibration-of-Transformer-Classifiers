"""Tests for src/data/cf_dataset.py."""
import random

import pytest
import torch
from transformers import AutoTokenizer

from src.data.cf_dataset import CounterfactualDataset, cf_collate_fn
from src.data.counterfactual_swap import CounterfactualSwapEngine


@pytest.fixture(scope="module")
def tokenizer():
    # Cheap tokenizer for tests; bert-base-uncased is widely cached.
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def deterministic_engine():
    return CounterfactualSwapEngine(
        swaps_per_token=2,
        pairs_cap_per_example=3,
        rng=random.Random(42),
    )


@pytest.fixture
def small_dataset(tokenizer, deterministic_engine):
    texts = [
        "I respect Jewish people",          # eligible: 2-3 swaps
        "Hello world, no identity here",    # no swaps
        "kike behavior is bad",             # slur → no swaps
        "Jewish leaders gathered today",    # eligible
    ]
    labels = [0, 0, 1, 0]
    return CounterfactualDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        swap_engine=deterministic_engine,
        max_length=32,
        return_source_group=True,
    )


class TestCounterfactualDataset:
    def test_len_matches_texts(self, small_dataset):
        assert len(small_dataset) == 4

    def test_getitem_no_identity_yields_zero_cf(self, small_dataset):
        item = small_dataset[1]
        assert item["cf_count"] == 0
        assert item["cf_input_ids"] == []
        assert item["cf_attention_masks"] == []

    def test_getitem_slur_yields_zero_cf(self, small_dataset):
        item = small_dataset[2]
        assert item["cf_count"] == 0

    def test_getitem_eligible_yields_cf(self, small_dataset):
        item = small_dataset[0]
        assert item["cf_count"] >= 1
        assert len(item["cf_input_ids"]) == item["cf_count"]
        for ids, am in zip(item["cf_input_ids"], item["cf_attention_masks"]):
            assert ids.shape == (32,)
            assert am.shape == (32,)
            assert ids.dtype == torch.long

    def test_source_group_included(self, small_dataset):
        item = small_dataset[0]
        assert "source_group_idx" in item
        assert item["source_group_idx"].dtype == torch.long

    def test_source_group_excluded_when_disabled(self, tokenizer, deterministic_engine):
        ds = CounterfactualDataset(
            texts=["jewish people"],
            labels=[0],
            tokenizer=tokenizer,
            swap_engine=deterministic_engine,
            max_length=16,
            return_source_group=False,
        )
        item = ds[0]
        assert "source_group_idx" not in item


class TestCfCollateFn:
    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            cf_collate_fn([])

    def test_pads_to_max_cf(self, small_dataset):
        # Indices 0 and 3 have CFs; indices 1 and 2 don't
        batch = [small_dataset[i] for i in range(4)]
        out = cf_collate_fn(batch)

        # Standard fields
        assert out["input_ids"].shape == (4, 32)
        assert out["attention_mask"].shape == (4, 32)
        assert out["labels"].shape == (4,)
        assert out["source_group_idx"].shape == (4,)

        # CF fields padded to max K
        max_cf = max(b["cf_count"] for b in batch)
        assert out["cf_input_ids"].shape == (4, max_cf, 32)
        assert out["cf_attention_mask"].shape == (4, max_cf, 32)
        assert out["cf_mask"].shape == (4, max_cf)
        assert out["cf_mask"].dtype == torch.bool

        # Mask correctness: row i should have b['cf_count'] True entries
        for i, b in enumerate(batch):
            assert out["cf_mask"][i].sum().item() == b["cf_count"]

    def test_zero_cf_in_whole_batch(self, tokenizer, deterministic_engine):
        ds = CounterfactualDataset(
            texts=["hello world", "good morning"],
            labels=[0, 0],
            tokenizer=tokenizer,
            swap_engine=deterministic_engine,
            max_length=16,
        )
        batch = [ds[0], ds[1]]
        out = cf_collate_fn(batch)
        assert out["cf_input_ids"].shape == (2, 0, 16)
        assert out["cf_mask"].shape == (2, 0)
        assert out["cf_mask"].sum().item() == 0
