"""PyTorch Dataset for GoldStandard2024."""
import random
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.data.preprocessing import mask_keywords, ANTISEMITISM_KEYWORDS


class AntisemitismDataset(Dataset):
    """
    PyTorch Dataset for antisemitism detection.

    Supports optional keyword masking augmentation (for Model C).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        keyword_masking: bool = False,
        keyword_mask_prob: float = 0.3,
        keywords: Optional[list[str]] = None,
        label_mask: Optional[list[bool]] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword_masking = keyword_masking
        self.keyword_mask_prob = keyword_mask_prob
        self.keywords = keywords or ANTISEMITISM_KEYWORDS
        # Semi-supervised setting: rows outside the mask are encoded and used by
        # CCI's consistency term but excluded from the supervised loss. Default
        # None means fully supervised, and __getitem__ then emits no extra key,
        # so existing collate paths are untouched.
        if label_mask is not None and len(label_mask) != len(texts):
            raise ValueError(
                f"label_mask has {len(label_mask)} entries for {len(texts)} texts"
            )
        self.label_mask = label_mask

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        # Apply keyword masking augmentation (training only).
        # Use the tokenizer's own mask token (DeBERTa: "[MASK]", RoBERTa: "<mask>").
        # Hardcoding "[MASK]" broke RoBERTa: the literal string was BPE-split into
        # 4-5 garbage subtokens instead of mapping to the single mask token id.
        if self.keyword_masking and random.random() < self.keyword_mask_prob:
            text = mask_keywords(text, self.keywords, mask_token=self.tokenizer.mask_token)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        if self.label_mask is not None:
            item["label_mask"] = torch.tensor(
                bool(self.label_mask[idx]), dtype=torch.bool
            )
        return item
