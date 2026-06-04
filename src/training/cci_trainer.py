"""Trainer subclass for CCI v2: focal + λ·CCI loss with diagnostic instrumentation.

Mirrors :class:`src.training.clp_trainer.CLPTrainer` but adds:

  * Joint optimization of the model + a :class:`TemperatureHead` (the head's
    parameters are added to the same AdamW optimizer with the same LR as the
    rest of the model).
  * Per-batch propagation of ``source_group_ids`` from the
    :class:`CounterfactualDataset` into the CCI loss.
  * Per-epoch recording of ``T_g`` to a :class:`TemperatureTrajectory`, which
    is dumped as JSON next to the checkpoint at end of training.

Expects the train loader to yield batches produced by ``cf_collate_fn``,
i.e. with ``cf_input_ids``, ``cf_attention_mask``, ``cf_mask`` and
``source_group_idx``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.temperature_head import TemperatureHead
from src.training.cci_diagnostics import TemperatureTrajectory
from src.training.cci_loss import CCIv2Loss
from src.training.trainer import Trainer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CCITrainer(Trainer):
    """Focal + λ·CCIv2 trainer with per-group T learning.

    Parameters
    ----------
    temperature_head : TemperatureHead
        Owned by the trainer. Its parameters are added to the AdamW optimizer
        if it is learnable; ignored otherwise.
    cci_loss_fn : CCIv2Loss
        The CCI loss instance (constructed by the caller with the same head).
    lambda_cci : float
        Weight on the CCI penalty. ``0.0`` reduces this trainer to plain focal.
    All other arguments are forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        temperature_head: TemperatureHead,
        cci_loss_fn: CCIv2Loss,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        checkpoint_strategy=None,
        lambda_cci: float = 1.0,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            config=config,
            device=device,
            checkpoint_strategy=checkpoint_strategy,
        )
        self.temperature_head = temperature_head.to(device)
        self.cci_loss_fn = cci_loss_fn
        self.lambda_cci = lambda_cci

        # If the head has parameters (learnable variant), tack them onto the
        # existing AdamW optimizer so they're updated jointly with the model.
        head_params = [p for p in self.temperature_head.parameters() if p.requires_grad]
        if head_params:
            # weight_decay=0 on log_T (L2 in log space is meaningless).
            self.optimizer.add_param_group({
                "params": head_params,
                "lr": config["learning_rate"],
                "weight_decay": 0.0,
            })
            # Rebuild scheduler: LambdaLR caches base_lrs at construction,
            # so the head group added above is otherwise frozen at init lr.
            import math
            effective_steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
            total_steps = effective_steps_per_epoch * self.config["epochs"]
            warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(
                "TemperatureHead is learnable: added %d log_T parameters to optimizer "
                "(lr=%.2e, weight_decay=0) and rebuilt linear-warmup scheduler "
                "with %d total optimizer steps (%d warmup).",
                len(head_params), config["learning_rate"], total_steps, warmup_steps,
            )
        else:
            logger.info("TemperatureHead is fixed (buffer); no extra optim params.")

        # Per-epoch T_g trajectory for the kill-criterion diagnostic.
        self.temperature_trajectory = TemperatureTrajectory()

        # Per-epoch component-loss accumulators
        self._epoch_focal_sum = 0.0
        self._epoch_cci_sum = 0.0
        self._epoch_n_batches = 0
        self._epoch_n_batches_with_cf = 0

    def train_epoch(self) -> float:
        """One epoch with focal + λ·CCI."""
        self.model.train()
        self.temperature_head.train()
        total_loss = 0.0
        self._epoch_focal_sum = 0.0
        self._epoch_cci_sum = 0.0
        self._epoch_n_batches = 0
        self._epoch_n_batches_with_cf = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training (CCI v2)", leave=False)
        for step, batch in enumerate(pbar):
            batch = self._move_batch_to_device(batch)

            with torch.amp.autocast(
                "cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                focal_loss = outputs["loss"]
                logits_orig = outputs["logits"]

                cci_loss = self._compute_cci_loss(batch, logits_orig=logits_orig)

                loss = focal_loss + self.lambda_cci * cci_loss
                loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            self._epoch_focal_sum += focal_loss.item()
            self._epoch_cci_sum += cci_loss.item()
            self._epoch_n_batches += 1
            if batch["cf_mask"].any().item():
                self._epoch_n_batches_with_cf += 1

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                # Clip across BOTH the backbone and the temperature head so a
                # noisy gradient on log_T cannot blow up training. Building the
                # combined list per step is cheap (head has ≤ num_groups
                # parameters) and avoids any need to remember the cap separately.
                clip_params = (
                    list(self.model.parameters())
                    + list(self.temperature_head.parameters())
                )
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        clip_params, self.config.get("max_grad_norm", 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        clip_params, self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            pbar.set_postfix(
                focal=f"{focal_loss.item():.4f}",
                cci=f"{cci_loss.item():.4f}",
            )

        # Per-epoch summary + T_g recording
        T_all = self.temperature_head.all_temperatures().detach()
        n_batches = max(1, self._epoch_n_batches)
        n_with_cf = max(1, self._epoch_n_batches_with_cf)
        if wandb.run is not None:
            wandb.log({
                "train_focal_loss": self._epoch_focal_sum / n_batches,
                "train_cci_loss": self._epoch_cci_sum / n_with_cf,
                "train_pct_batches_with_cf": self._epoch_n_batches_with_cf / n_batches,
                **{
                    f"T_{name}": float(T_all[i].item())
                    for i, name in enumerate(self.temperature_trajectory.group_names)
                    if i < T_all.shape[0]
                },
            })
        logger.info(
            "Epoch CCI summary: focal=%.4f cci=%.4f T=%s",
            self._epoch_focal_sum / n_batches,
            self._epoch_cci_sum / n_with_cf,
            "[" + ", ".join(f"{t:.3f}" for t in T_all.cpu().tolist()) + "]",
        )

        return total_loss / len(self.train_loader)

    def train(self, save_dir: str) -> dict:
        """Wrap the parent train() to record T_g at every epoch end."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Hook into parent's train() — but the parent doesn't expose an epoch
        # callback, so we re-implement the loop with the recording line added.
        # We replicate the parent's behaviour exactly and just add T_g logging.
        history = {"train_loss": [], "dev_metrics": []}
        logger.info(
            "Starting CCI v2 training: %d epochs, batch_size=%s, grad_accum=%d, "
            "device=%s, precision=%s, lambda_cci=%.3f",
            self.config["epochs"],
            self.config.get("batch_size", "?"),
            self.grad_accum_steps,
            self.device,
            self.precision,
            self.lambda_cci,
        )

        import time

        for epoch in range(self.config["epochs"]):
            start = time.time()
            train_loss = self.train_epoch()
            dev_metrics = self.evaluate()
            elapsed = time.time() - start

            # Record T_g for the diagnostic
            self.temperature_trajectory.record(
                epoch_idx=epoch,
                T_per_group=self.temperature_head.all_temperatures(),
            )

            logger.info(
                "Epoch %d/%d | Loss: %.4f | Dev F1: %.4f | Dev Prec: %.4f | "
                "Dev Rec: %.4f | Time: %.0fs",
                epoch + 1, self.config["epochs"], train_loss,
                dev_metrics["macro_f1"], dev_metrics["precision"],
                dev_metrics["recall"], elapsed,
            )

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    **{f"dev_{k}": v for k, v in dev_metrics.items()
                       if isinstance(v, (int, float))},
                })

            history["train_loss"].append(train_loss)
            history["dev_metrics"].append(dev_metrics)

            self.checkpoint_strategy.on_epoch_end(
                epoch=epoch,
                model=self.model,
                dev_metrics=dev_metrics,
                dev_loader=self.dev_loader,
                device=self.device,
            )

            score_name = self.checkpoint_strategy.name
            current_metric = self.checkpoint_strategy.score(dev_metrics)
            # Strict > to match parent Trainer's convention; ties do not reset patience.
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.best_checkpoint_path = save_path / "best_model.pt"
                # Save BOTH the model state and the temperature head state.
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "temperature_head": self.temperature_head.state_dict(),
                    },
                    self.best_checkpoint_path,
                )
                logger.info(f"New best checkpoint saved ({score_name}={current_metric:.4f})")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement ({score_name}={current_metric:.4f} vs "
                    f"best={self.best_metric:.4f}), patience={self.patience_counter}/{self.patience}"
                )
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            ckpt = torch.load(
                self.best_checkpoint_path, weights_only=True, map_location=self.device
            )
            self.model.load_state_dict(ckpt["model"])
            self.temperature_head.load_state_dict(ckpt["temperature_head"])
            logger.info("Loaded best checkpoint (model + temperature head).")

        # Strategy finalize is unchanged
        self.model = self.checkpoint_strategy.finalize(
            model=self.model,
            save_dir=save_path,
            dev_loader=self.dev_loader,
            device=self.device,
        )

        history["dev_metrics_after_finalize"] = self.evaluate()
        history["checkpoint_strategy_state"] = self.checkpoint_strategy.export_state()

        # Persist the temperature trajectory for the kill-criterion analysis.
        self.temperature_trajectory.to_json(save_path / "temperature_trajectory.json")
        history["temperature_trajectory"] = self.temperature_trajectory.to_dict()

        return history

    # ------------------------------------------------------------------ helpers

    def _move_batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_cci_loss(
        self,
        batch: dict,
        logits_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Forward on the (B, K, L) counterfactual block + apply CCI."""
        cf_mask: torch.Tensor = batch["cf_mask"]
        if cf_mask.numel() == 0 or not cf_mask.any().item():
            return logits_orig.new_zeros(())

        cf_input_ids = batch["cf_input_ids"]
        cf_attention_mask = batch["cf_attention_mask"]
        B, K, L = cf_input_ids.shape

        cf_outputs = self.model(
            input_ids=cf_input_ids.view(B * K, L),
            attention_mask=cf_attention_mask.view(B * K, L),
        )
        cf_logits = cf_outputs["logits"].view(B, K, -1)

        if "source_group_idx" not in batch:
            raise KeyError(
                "CCITrainer expects batch['source_group_idx'] to be present. "
                "Did you build the dataset with return_source_group=True?"
            )
        source_group_ids = batch["source_group_idx"]

        return self.cci_loss_fn(logits_orig, cf_logits, cf_mask, source_group_ids)
