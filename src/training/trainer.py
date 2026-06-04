"""Training loop with early stopping, gradient accumulation, and logging."""
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from src.evaluation.metrics import compute_metrics
from src.training.checkpoint_strategies import (
    CheckpointStrategy,
    BestF1Strategy,
    build_strategy,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training loop for transformer classifiers.

    Features:
    - Mixed precision training (fp16/bf16/fp32), device-agnostic autocast,
      GradScaler only on fp16 (bf16 has fp32 dynamic range)
    - Gradient accumulation for effective larger batch sizes on small GPUs
    - Linear warmup + decay schedule
    - Early stopping on dev macro-F1
    - WandB logging
    - Checkpoint saving
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        checkpoint_strategy: Optional[CheckpointStrategy] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.device = device
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)

        # Checkpoint-selection strategy: defaults to BestF1 (backward-compat).
        # If config has training.checkpoint_strategy, instantiate from there;
        # otherwise the explicit constructor argument wins; otherwise BestF1.
        if checkpoint_strategy is None:
            spec = config.get("checkpoint_strategy")
            checkpoint_strategy = build_strategy(spec) if spec is not None else BestF1Strategy()
        self.checkpoint_strategy = checkpoint_strategy

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": config.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_params, lr=config["learning_rate"])

        # Scheduler — account for gradient accumulation, including the partial
        # flush on the final step of each epoch (the train loop steps when
        # step+1 == len(loader) even if not divisible by grad_accum).
        import math
        effective_steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        total_steps = effective_steps_per_epoch * config["epochs"]
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision — resolve to one of {"fp32", "fp16", "bf16"}.
        # Accepts `precision: bf16|fp16|fp32` (new) or legacy `fp16: true/false`.
        precision = config.get("precision")
        if precision is None:
            precision = "fp16" if config.get("fp16", True) else "fp32"
        precision = precision.lower()
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unknown precision {precision!r}; expected fp32/fp16/bf16")
        if device == "cpu":
            precision = "fp32"  # autocast/GradScaler are CUDA-only here
        self.precision = precision
        self.autocast_enabled = precision in {"fp16", "bf16"}
        self.autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision]
        # GradScaler is only needed for fp16 (bf16 has fp32 dynamic range).
        self.scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

        # Early stopping
        self.patience = config.get("patience", 3)
        # ``best_metric`` was previously a fixed metric (macro_f1); now it is
        # whatever ``self.checkpoint_strategy.score`` returns. Some strategies
        # (composite F1 - alpha*ECE) can be negative early in training, so we
        # initialise to -inf.
        self.best_metric = float("-inf")
        self.patience_counter = 0
        self.best_checkpoint_path = None

    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                "cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps

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

            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum_steps:.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> dict:
        """Evaluate on dev (or any) loader. Returns metrics dict.

        Logits are passed through ``self.checkpoint_strategy.transform_logits``
        before softmax. For most strategies this is the identity; only
        TemperatureScaling rescales them by the fitted T (and only after
        ``finalize`` has been called)."""
        self.model.eval()
        loader = loader or self.dev_loader

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                "cuda", dtype=self.autocast_dtype, enabled=self.autocast_enabled
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            transformed = self.checkpoint_strategy.transform_logits(outputs["logits"])
            probs = torch.softmax(transformed, dim=-1)
            preds = transformed.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics

    def train(self, save_dir: str) -> dict:
        """Full training loop with early stopping."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "dev_metrics": []}

        logger.info(
            f"Starting training: {self.config['epochs']} epochs, "
            f"batch_size={self.config.get('batch_size', '?')}, "
            f"grad_accum={self.grad_accum_steps}, "
            f"effective_batch={self.config.get('batch_size', 0) * self.grad_accum_steps}, "
            f"device={self.device}, precision={self.precision}"
        )

        for epoch in range(self.config["epochs"]):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Evaluate
            dev_metrics = self.evaluate()

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Loss: {train_loss:.4f} | "
                f"Dev F1: {dev_metrics['macro_f1']:.4f} | "
                f"Dev Prec: {dev_metrics['precision']:.4f} | "
                f"Dev Rec: {dev_metrics['recall']:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    **{f"dev_{k}": v for k, v in dev_metrics.items()
                       if isinstance(v, (int, float))},
                })

            history["train_loss"].append(train_loss)
            history["dev_metrics"].append(dev_metrics)

            # Strategy hook: SWA snapshots, AdaFocal gamma update, etc.
            self.checkpoint_strategy.on_epoch_end(
                epoch=epoch,
                model=self.model,
                dev_metrics=dev_metrics,
                dev_loader=self.dev_loader,
                device=self.device,
            )

            # Early stopping check — the strategy decides what "better" means.
            score_name = self.checkpoint_strategy.name
            current_metric = self.checkpoint_strategy.score(dev_metrics)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.best_checkpoint_path = save_path / "best_model.pt"
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                logger.info(
                    f"New best model saved ({score_name}={current_metric:.4f})"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement ({score_name}={current_metric:.4f} vs "
                    f"best={self.best_metric:.4f}), patience={self.patience_counter}/{self.patience}"
                )
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best checkpoint
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(self.best_checkpoint_path, weights_only=True, map_location=self.device)
            )
            logger.info("Loaded best checkpoint for final evaluation.")

        # Strategy finalize: TS fits T on dev, SWA averages snapshot weights.
        self.model = self.checkpoint_strategy.finalize(
            model=self.model,
            save_dir=save_path,
            dev_loader=self.dev_loader,
            device=self.device,
        )

        # Re-evaluate dev after finalize so the returned history reflects
        # post-hoc transforms (TS-rescaled logits, SWA-averaged weights).
        history["dev_metrics_after_finalize"] = self.evaluate()
        history["checkpoint_strategy_state"] = self.checkpoint_strategy.export_state()

        return history
