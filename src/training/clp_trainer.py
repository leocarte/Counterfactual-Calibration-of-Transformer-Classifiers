"""Trainer subclass that adds Counterfactual Logit Pairing to the focal loss.

Subclasses :class:`src.training.trainer.Trainer` so all the existing machinery
(checkpoint strategies, mixed precision, gradient accumulation, early stopping,
WandB logging) is preserved. Only ``train_epoch`` is overridden — evaluation
remains identical (CLP is a training-time term only).

Expects the train loader to yield batches produced by
:func:`src.data.cf_dataset.cf_collate_fn`, i.e. with the extra fields
``cf_input_ids``, ``cf_attention_mask``, ``cf_mask``.
"""
from __future__ import annotations

from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.clp_loss import CLPLoss
from src.training.trainer import Trainer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CLPTrainer(Trainer):
    """Trainer with Counterfactual Logit Pairing (Garg AIES 2019) added to focal loss.

    Parameters
    ----------
    lambda_clp : float
        Weight on the CLP penalty. ``0.0`` reduces this trainer to plain focal
        training (useful as an ablation). Sweep range in our setting:
        ``{0.5, 1.0, 2.0}``.
    clp_loss_fn : CLPLoss, optional
        Custom CLP loss instance. Defaults to ``CLPLoss(divergence='l1')``,
        which matches Garg's original formulation.
    All other arguments are forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        checkpoint_strategy=None,
        lambda_clp: float = 1.0,
        clp_loss_fn: Optional[CLPLoss] = None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            config=config,
            device=device,
            checkpoint_strategy=checkpoint_strategy,
        )
        self.lambda_clp = lambda_clp
        self.clp_loss_fn = clp_loss_fn or CLPLoss(divergence="l1")
        # Per-epoch component-loss accumulators (logged to WandB)
        self._epoch_focal_sum = 0.0
        self._epoch_clp_sum = 0.0
        self._epoch_n_batches = 0
        self._epoch_n_batches_with_cf = 0

    def train_epoch(self) -> float:
        """One epoch with focal + λ·CLP loss.

        Returns the average *total* loss across batches (matches parent
        :meth:`Trainer.train_epoch` contract). Per-component breakdown is
        accessible via the instance attributes ``self._epoch_focal_sum`` etc.,
        which the overridden ``train`` logs to WandB.
        """
        self.model.train()
        total_loss = 0.0
        self._epoch_focal_sum = 0.0
        self._epoch_clp_sum = 0.0
        self._epoch_n_batches = 0
        self._epoch_n_batches_with_cf = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training (CLP)", leave=False)
        for step, batch in enumerate(pbar):
            batch = self._move_batch_to_device(batch)

            with torch.amp.autocast(
                "cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled
            ):
                # Forward on the original example — produces focal loss
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                focal_loss = outputs["loss"]
                logits_orig = outputs["logits"]  # (B, C)

                # CLP forward on counterfactuals (only if the batch has any)
                clp_loss = self._compute_clp_loss(
                    batch, logits_orig=logits_orig,
                )

                loss = focal_loss + self.lambda_clp * clp_loss
                loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            self._epoch_focal_sum += focal_loss.item()
            self._epoch_clp_sum += clp_loss.item()
            self._epoch_n_batches += 1
            if batch["cf_mask"].any().item():
                self._epoch_n_batches_with_cf += 1

            # Optimizer step every grad_accum_steps
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            pbar.set_postfix(
                focal=f"{focal_loss.item():.4f}",
                clp=f"{clp_loss.item():.4f}",
                total=f"{(focal_loss.item() + self.lambda_clp * clp_loss.item()):.4f}",
            )

        # Per-epoch summary (logged to WandB by parent train() via dev_metrics)
        if wandb.run is not None and self._epoch_n_batches > 0:
            wandb.log({
                "train_focal_loss": self._epoch_focal_sum / self._epoch_n_batches,
                "train_clp_loss": (
                    self._epoch_clp_sum / max(1, self._epoch_n_batches_with_cf)
                ),
                "train_pct_batches_with_cf": (
                    self._epoch_n_batches_with_cf / self._epoch_n_batches
                ),
            })
        logger.info(
            f"Epoch CLP summary: focal={self._epoch_focal_sum / self._epoch_n_batches:.4f}, "
            f"clp={self._epoch_clp_sum / max(1, self._epoch_n_batches_with_cf):.4f}, "
            f"%batches with cf pairs={self._epoch_n_batches_with_cf / self._epoch_n_batches:.1%}"
        )

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------ helpers

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all torch.Tensor entries in the batch dict to ``self.device``."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_clp_loss(
        self,
        batch: dict,
        logits_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model on the (B, K, L) counterfactual block and apply CLP.

        Returns a zero scalar with the right dtype/device if the batch has
        no valid counterfactual pairs (the ``CLPLoss`` short-circuit also
        handles this, but we avoid the unnecessary forward pass here).
        """
        cf_mask: torch.Tensor = batch["cf_mask"]  # (B, K)
        if cf_mask.numel() == 0 or not cf_mask.any().item():
            return logits_orig.new_zeros(())

        cf_input_ids = batch["cf_input_ids"]      # (B, K, L)
        cf_attention_mask = batch["cf_attention_mask"]  # (B, K, L)
        B, K, L = cf_input_ids.shape

        # Flatten (B, K) → (B*K) so we can call the model once.
        cf_outputs = self.model(
            input_ids=cf_input_ids.view(B * K, L),
            attention_mask=cf_attention_mask.view(B * K, L),
        )
        cf_logits = cf_outputs["logits"].view(B, K, -1)  # (B, K, C)

        return self.clp_loss_fn(logits_orig, cf_logits, cf_mask)
