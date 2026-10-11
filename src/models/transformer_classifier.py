"""Transformer-based classifier for antisemitism detection."""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.focal_loss import FocalLoss


class TransformerClassifier(nn.Module):
    """
    Wrapper around HuggingFace AutoModelForSequenceClassification.

    Supports:
    - Any pretrained transformer (RoBERTa, DeBERTa, HateBERT)
    - Weighted cross-entropy or focal loss
    - Dropout configuration
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        loss_type: str = "weighted_ce",
        class_weights: Optional[list[float]] = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        super().__init__()

        # Force fp32 master weights. Some Hub checkpoints (e.g. microsoft/deberta-v3-base)
        # are stored in fp16; loading them as-is causes NaN under autocast and
        # "expected Half but found Float" in loss functions with fp32 buffers.
        # Autocast still downcasts activations to bf16/fp16 at runtime when enabled.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )

        # Loss function
        self.loss_type = loss_type
        if loss_type == "weighted_ce":
            weights = torch.tensor(class_weights or [1.0, 1.0], dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass; ``label_mask`` marks which rows carry a usable label.

        The mask exists for the semi-supervised setting, where part of the
        training pool is treated as unlabelled: those rows must still be encoded
        (CCI's consistency term applies to them) but must not reach the
        supervised loss. Unlabelled rows keep a dummy label so the one-hot inside
        FocalLoss stays in range; their per-example loss is computed and then
        discarded. ``label_mask=None`` reproduces the previous behaviour exactly.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        result = {"logits": logits}

        if labels is not None:
            # Move loss weights to same device as logits
            if hasattr(self.loss_fn, "weight") and self.loss_fn.weight is not None:
                self.loss_fn.weight = self.loss_fn.weight.to(logits.device)

            if label_mask is None:
                result["loss"] = self.loss_fn(logits, labels)
            else:
                # Average over labelled rows only. Switching reduction rather than
                # holding a second loss object keeps one source of truth for
                # gamma/alpha/weights; it is restored even if the loss raises.
                prev = self.loss_fn.reduction
                self.loss_fn.reduction = "none"
                try:
                    per_example = self.loss_fn(logits, labels)
                finally:
                    self.loss_fn.reduction = prev
                m = label_mask.to(per_example.dtype)
                # clamp: at small label fractions a batch can be entirely
                # unlabelled, and 0/0 would poison the gradient with NaN.
                result["loss"] = (per_example * m).sum() / m.sum().clamp(min=1.0)

        return result

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        return outputs["logits"].argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        return torch.softmax(outputs["logits"], dim=-1)

    @classmethod
    def load_tokenizer(cls, model_name: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(model_name)
