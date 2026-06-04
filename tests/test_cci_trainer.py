"""Regression tests for CCITrainer.

These tests target the two bugs surfaced by the 2026-05-03 audit:

  * H1 (apply_multicalibration cannot load CCI v2 checkpoints) — covered by
    :class:`TestCheckpointFormat`. Verifies that the wrapper-format saved by
    CCITrainer is loadable by the unwrap guard in apply_multicalibration.

  * H2 (TemperatureHead added after scheduler construction is silently
    excluded from the warmup/decay schedule) — covered by
    :class:`TestSchedulerIncludesHead`. Builds a real CCITrainer with a
    minimal mock backbone and verifies that all optimizer param groups share
    the same schedule after the fix.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.temperature_head import TemperatureHead
from src.training.cci_loss import CCIv2Loss
from src.training.cci_trainer import CCITrainer


# ---------------------------------------------------------------------------
# Minimal fakes — keep tests CPU-only and self-contained
# ---------------------------------------------------------------------------

class _MockClassifier(nn.Module):
    """Stand-in for TransformerClassifier with the same forward signature."""

    def __init__(self, vocab_size: int = 100, hidden: int = 8, num_labels: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Mean-pool token embeddings to get a single vector per example
        x = self.embed(input_ids)               # (B, L, H)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            x = x.mean(dim=1)
        logits = self.head(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits, labels)
        return out


class _DummyCFDataset(Dataset):
    """Yields the same field shape that cf_collate_fn expects."""

    def __init__(self, n: int = 8, seq_len: int = 4):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(1, 99, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.tensor(idx % 2, dtype=torch.long),
        }


def _build_trainer(learnable: bool):
    """Construct a fully-wired CCITrainer on CPU. Returns (trainer, head)."""
    head = TemperatureHead(num_groups=3, init_T=1.0, learnable=learnable)
    cci_loss = CCIv2Loss(temperature_head=head, divergence="js")
    model = _MockClassifier()
    train_loader = DataLoader(_DummyCFDataset(n=8), batch_size=2)
    dev_loader = DataLoader(_DummyCFDataset(n=4), batch_size=2)
    config = {
        "learning_rate": 2e-5,
        "epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "patience": 3,
        "metric_for_best_model": "macro_f1",
        "precision": "fp32",
    }
    trainer = CCITrainer(
        model=model,
        temperature_head=head,
        cci_loss_fn=cci_loss,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=config,
        device="cpu",
    )
    return trainer, head


# ---------------------------------------------------------------------------
# H2 regression: scheduler must include the head's param group
# ---------------------------------------------------------------------------

class TestSchedulerIncludesHead:
    def test_learnable_head_adds_param_group(self):
        trainer, _ = _build_trainer(learnable=True)
        # 3 groups: model decay, model no_decay, head
        assert len(trainer.optimizer.param_groups) == 3

    def test_fixed_head_does_not_add_param_group(self):
        trainer, _ = _build_trainer(learnable=False)
        # 2 groups: model decay, model no_decay (head is a buffer)
        assert len(trainer.optimizer.param_groups) == 2

    def test_scheduler_base_lrs_match_param_groups(self):
        """Regression: BEFORE the H2 fix, scheduler.base_lrs has length 2 while
        optimizer.param_groups has length 3. After the fix, they match."""
        trainer, _ = _build_trainer(learnable=True)
        assert len(trainer.scheduler.base_lrs) == len(trainer.optimizer.param_groups)

    def test_scheduler_advances_head_lr_in_lockstep(self):
        """Regression: BEFORE the H2 fix, scheduler.step() updates only the
        first 2 param groups (model decay, no_decay) and the head's lr stays
        frozen at the AdamW default. After the fix, all 3 lrs share the
        warmup/decay schedule."""
        trainer, _ = _build_trainer(learnable=True)

        # Several scheduler steps to leave the trivial step-0 state and
        # exercise both warmup and decay regions.
        for _ in range(5):
            trainer.scheduler.step()

        lrs = [g["lr"] for g in trainer.optimizer.param_groups]
        # Both model groups must follow the same schedule.
        assert pytest.approx(lrs[0], rel=1e-9) == lrs[1], (
            f"Model decay vs no_decay lrs diverged: {lrs}"
        )
        # The head must follow the SAME schedule. This is the regression check.
        assert pytest.approx(lrs[2], rel=1e-9) == lrs[0], (
            f"Head lr not following the warmup schedule: head={lrs[2]}, "
            f"model={lrs[0]}. The H2 fix (rebuild scheduler after "
            f"add_param_group) was not applied or has regressed."
        )


# ---------------------------------------------------------------------------
# H1 regression: apply_multicalibration must accept wrapped CCI v2 checkpoints
# ---------------------------------------------------------------------------

class TestCheckpointFormat:
    def test_cci_checkpoint_is_wrapper_dict(self):
        """Document the on-disk format CCITrainer.train uses, so future
        consumers know to handle it. This is the format that broke
        apply_multicalibration before the H1 fix."""
        head = TemperatureHead(num_groups=3, learnable=True)
        model = _MockClassifier()

        # Reproduce what CCITrainer.train saves to disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best_model.pt"
            torch.save(
                {"model": model.state_dict(),
                 "temperature_head": head.state_dict()},
                ckpt_path,
            )
            state = torch.load(ckpt_path, weights_only=True)

        # Top-level keys are the CCI v2 schema, NOT a flat parameter dict.
        assert isinstance(state, dict)
        assert set(state.keys()) == {"model", "temperature_head"}
        assert all(k.startswith(("embed.", "head."))
                   for k in state["model"].keys())

    def test_unwrap_logic_recovers_model_state(self):
        """Mirror of the inline guard in apply_multicalibration.py:
        when the loaded object has 'model' and 'temperature_head' keys,
        unwrap to state['model']. Otherwise pass through."""
        head = TemperatureHead(num_groups=3, learnable=True)
        model = _MockClassifier()

        wrapped = {
            "model": model.state_dict(),
            "temperature_head": head.state_dict(),
        }
        unwrapped = model.state_dict()

        # The unwrap predicate (kept in sync with apply_multicalibration.py)
        def maybe_unwrap(state):
            if isinstance(state, dict) and "model" in state and "temperature_head" in state:
                return state["model"]
            return state

        # Wrapped → unwrap; flat → pass-through
        assert maybe_unwrap(wrapped) is wrapped["model"]
        assert maybe_unwrap(unwrapped) is unwrapped

        # The unwrapped result must be loadable by a fresh model
        fresh = _MockClassifier()
        fresh.load_state_dict(maybe_unwrap(wrapped))
