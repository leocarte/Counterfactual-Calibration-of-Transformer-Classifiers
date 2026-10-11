"""PyTorch Dataset that augments AntisemitismDataset with counterfactual pairs.

Each ``__getitem__`` call returns the standard fields plus a variable-length list
of tokenized counterfactuals (one per symmetry-filtered swap of the source text).
The companion :func:`cf_collate_fn` collates the variable-length lists into a
padded ``(B, K_max, L)`` tensor with a boolean mask.

Used by the CLP / CCI v2 trainers. The base ``AntisemitismDataset`` is
unaffected — non-CLP training continues to use it directly.

Also hosts :func:`expand_with_counterfactual_rows`, the static train-split
expansion behind the CDA baseline (counterfactual data augmentation: swapped
copies as extra labeled rows, no consistency loss).
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from src.data.counterfactual_swap import CounterfactualSwapEngine
from src.data.dataset import AntisemitismDataset
from src.data.identity_inventory import (
    group_label_to_index,
    primary_keyword_group,
)


class CounterfactualDataset(AntisemitismDataset):
    """AntisemitismDataset + online counterfactual generation.

    Parameters
    ----------
    swap_engine : CounterfactualSwapEngine
        Pre-configured generator. Symmetry filtering happens inside it.
    return_source_group : bool, default True
        If True, ``__getitem__`` includes ``source_group_idx`` (int tensor)
        identifying which CCI v2 keyword group the source text belongs to.
        Required by CCI v2's TemperatureHead; harmless for plain Davani.
    All other parameters are forwarded to :class:`AntisemitismDataset`.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        swap_engine: CounterfactualSwapEngine,
        max_length: int = 128,
        keyword_masking: bool = False,
        keyword_mask_prob: float = 0.3,
        keywords: Optional[list[str]] = None,
        return_source_group: bool = True,
        label_mask: Optional[list[bool]] = None,
    ):
        super().__init__(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
            keyword_masking=keyword_masking,
            keyword_mask_prob=keyword_mask_prob,
            keywords=keywords,
            label_mask=label_mask,
        )
        self.swap_engine = swap_engine
        self.return_source_group = return_source_group

    def __getitem__(self, idx: int) -> dict:
        # Original-text encoding (handles keyword masking augmentation if on)
        item = super().__getitem__(idx)

        text = self.texts[idx]
        # Keyword-masking augmentation in the parent class would have produced
        # a different encoding; for CLP pairing we always swap *the original
        # text* (un-augmented) so the pair semantics are clean.
        cf_pairs = self.swap_engine.generate(text)

        cf_input_ids: list[torch.Tensor] = []
        cf_attention_masks: list[torch.Tensor] = []
        for pair in cf_pairs:
            enc = self.tokenizer(
                pair.swap_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            cf_input_ids.append(enc["input_ids"].squeeze(0))
            cf_attention_masks.append(enc["attention_mask"].squeeze(0))

        item["cf_input_ids"] = cf_input_ids  # list of (L,) tensors, len = num pairs
        item["cf_attention_masks"] = cf_attention_masks
        item["cf_count"] = len(cf_pairs)

        if self.return_source_group:
            group = primary_keyword_group(text)
            item["source_group_idx"] = torch.tensor(
                group_label_to_index(group), dtype=torch.long
            )

        return item


def expand_with_counterfactual_rows(
    texts: list[str],
    labels: list[int],
    swap_engine: CounterfactualSwapEngine,
    exclude_texts: Optional[set[str]] = None,
) -> tuple[list[str], list[int]]:
    """Statically expand a split with counterfactual copies (the CDA baseline).

    Plain counterfactual data augmentation: every admissible identity swap of
    a source row is appended as an additional labeled row, with the label
    copied verbatim from the source. No pairing information is kept — the
    result feeds the plain :class:`AntisemitismDataset` and the standard
    :class:`~src.training.trainer.Trainer` (no consistency loss anywhere).

    Call this on the TRAIN split only. Dev/test must stay un-augmented: the
    model is always evaluated on real tweets, and the CFR/robustness metrics
    generate their own swaps at evaluation time.

    Parameters
    ----------
    texts, labels :
        The source split as parallel lists.
    swap_engine : CounterfactualSwapEngine
        Pre-seeded engine; symmetry filtering happens inside it. Configure it
        identically to the CLP/CCI trainers so CDA sees exactly the same
        admissible-swap distribution as the consistency-loss methods.
    exclude_texts : set[str], optional
        Texts that must never enter the augmented split — pass the union of
        dev and test texts. Guards against the corner case where a swapped
        train tweet becomes character-identical to an evaluation tweet.

    Returns
    -------
    (texts, labels)
        New lists with the augmented rows appended after the originals.
        Deterministic given a seeded engine. Swapped texts are deduplicated
        against the originals and against each other.
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels length mismatch: {len(texts)} vs {len(labels)}"
        )
    exclude = exclude_texts or set()
    seen = set(texts)  # never add a row whose text already exists in the split
    aug_texts: list[str] = []
    aug_labels: list[int] = []
    for text, label in zip(texts, labels):
        for pair in swap_engine.generate(text):
            swap_text = pair.swap_text
            if swap_text in seen or swap_text in exclude:
                continue
            seen.add(swap_text)
            aug_texts.append(swap_text)
            aug_labels.append(label)
    return list(texts) + aug_texts, list(labels) + aug_labels


def cf_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length counterfactual lists into padded tensors.

    Output shape:
      input_ids        (B, L)
      attention_mask   (B, L)
      labels           (B,)
      cf_input_ids     (B, K_max, L)        K_max = max cf_count in this batch
      cf_attention_mask(B, K_max, L)
      cf_mask          (B, K_max)           bool, True = real counterfactual
      source_group_idx (B,)                 if present

    If no example in the batch has any counterfactual, K_max defaults to 0 and
    the cf_* tensors are empty (shape (B, 0, L) / (B, 0)). The CLP loss
    short-circuits to zero in that case.
    """
    if not batch:
        raise ValueError("cf_collate_fn called with empty batch")

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    seq_len = input_ids.shape[1]
    max_cf = max(b["cf_count"] for b in batch)

    cf_input_ids = torch.zeros(
        (len(batch), max_cf, seq_len), dtype=input_ids.dtype
    )
    cf_attention_mask = torch.zeros(
        (len(batch), max_cf, seq_len), dtype=attention_mask.dtype
    )
    cf_mask = torch.zeros((len(batch), max_cf), dtype=torch.bool)

    for i, b in enumerate(batch):
        for j, (ids, am) in enumerate(zip(b["cf_input_ids"], b["cf_attention_masks"])):
            cf_input_ids[i, j] = ids
            cf_attention_mask[i, j] = am
            cf_mask[i, j] = True

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "cf_input_ids": cf_input_ids,
        "cf_attention_mask": cf_attention_mask,
        "cf_mask": cf_mask,
    }

    if "source_group_idx" in batch[0]:
        out["source_group_idx"] = torch.stack([b["source_group_idx"] for b in batch])

    # Present only in the semi-supervised setting; when absent the batch keys are
    # identical to the fully supervised path, so nothing downstream changes.
    if "label_mask" in batch[0]:
        out["label_mask"] = torch.stack([b["label_mask"] for b in batch])

    return out
