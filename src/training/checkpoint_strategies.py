"""Checkpoint-selection strategies for the calibration bake-off (Layer 1).

Background: under focal-loss training with macro-F1-based early stopping we
observed (commit eac5cf3 / paper §4.2) a bimodal test-ECE distribution across
seeds: 4/6 seeds converge to a well-calibrated regime (ECE ~0.05) while 2/6
peak F1 at epoch 3 and save an under-calibrated checkpoint (ECE ~0.13). To
characterise *this* phenomenon honestly and benchmark plausible fixes, we
implement five checkpoint-selection strategies as plug-in classes; the trainer
uses ``score`` for the early-stopping decision, ``on_epoch_end`` for any
training-time bookkeeping (e.g. SWA snapshots), ``finalize`` for any post-hoc
work (e.g. fitting a temperature parameter), and ``transform_logits`` to apply
inference-time logit transformations (only TS uses it).

Implemented here:
  - BestF1Strategy           : vanilla F1-on-dev early stopping (current default).
  - BestF1PositiveStrategy   : same but on the positive-class F1 (better for
                                imbalanced binary tasks; cf. Saito & Rehmsmeier 2015).
  - CompositeF1ECEStrategy   : argmax_e [F1(e) - alpha * ECE(e)]; a linear
                                scalarisation of the calibration/F1 Pareto frontier.
  - TemperatureScalingStrategy: vanilla F1-stop training, post-hoc T fitted on
                                dev to minimise NLL (Guo et al., ICML 2017).
  - SWAStrategy              : Stochastic Weight Averaging over the last K epochs
                                (Izmailov et al., UAI 2018). Score for early stop
                                stays F1; SWA averaging happens at finalise.

AdaFocal (Ghosh et al., NeurIPS 2022) lives in a separate module because it
requires modifying the loss-function's gamma during training, which couples
to the model rather than the checkpoint.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CheckpointStrategy:
    """Base class. Subclasses override one or more hooks.

    Defaults match the prior trainer behaviour (early-stop on macro-F1, no
    post-hoc transformation), so passing no strategy reproduces results
    pre-2026-04-27.
    """
    name: str = "base"

    def score(self, dev_metrics: dict) -> float:
        """Return a scalar — higher is better — used by the trainer for the
        early-stopping comparison (was: ``dev_metrics[metric_for_best_model]``)."""
        return dev_metrics["macro_f1"]

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        dev_metrics: dict,
        dev_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> None:
        """Hook called after every epoch. Default: no-op. SWA uses it."""
        return None

    def finalize(
        self,
        model: torch.nn.Module,
        save_dir: Path,
        dev_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> torch.nn.Module:
        """Called after training ends, after the best checkpoint is loaded.
        Returns the final model (possibly modified). Default: identity."""
        return model

    @torch.no_grad()
    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Inference-time logit transformation (only TS uses it). Default: identity."""
        return logits

    def export_state(self) -> dict:
        """Strategy-specific bookkeeping to log alongside metrics
        (alpha for composite, fitted_T for TS, snapshot epochs for SWA)."""
        return {"name": self.name}


# -------- score-only strategies --------------------------------------------


class BestF1Strategy(CheckpointStrategy):
    """Vanilla early stopping on macro-F1. Reproduces pre-pivot behaviour."""
    name = "best_f1"

    def score(self, dev_metrics: dict) -> float:
        return dev_metrics["macro_f1"]


class BestF1PositiveStrategy(CheckpointStrategy):
    """Early-stop on positive-class F1. Better motivated for our 4.79:1 task."""
    name = "best_f1_positive"

    def score(self, dev_metrics: dict) -> float:
        return dev_metrics["f1_positive"]


class CompositeF1ECEStrategy(CheckpointStrategy):
    """Score = F1 - alpha * ECE. Linear scalarisation of the calibration/F1
    Pareto frontier; alpha=0 collapses to BestF1, larger alpha trades F1 for
    calibration. We do not claim alpha as a tuned hyperparameter; the bake-off
    sweeps alpha in {0.5, 1.0, 2.0} on dev.
    """

    def __init__(self, alpha: float = 1.0, base_metric: str = "macro_f1"):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if base_metric not in {"macro_f1", "f1_positive"}:
            raise ValueError(f"base_metric must be macro_f1 or f1_positive, got {base_metric!r}")
        self.alpha = float(alpha)
        self.base_metric = base_metric
        self.name = f"composite_{base_metric}_ece_alpha{alpha:g}"

    def score(self, dev_metrics: dict) -> float:
        return dev_metrics[self.base_metric] - self.alpha * dev_metrics["ece"]

    def export_state(self) -> dict:
        return {"name": self.name, "alpha": self.alpha, "base_metric": self.base_metric}


# -------- post-hoc strategies (require finalize) ---------------------------


class TemperatureScalingStrategy(CheckpointStrategy):
    """Vanilla F1-stop during training. After loading the best checkpoint, fit
    a single temperature scalar T on the dev set by minimising NLL (Guo et al.,
    ICML 2017). Test-time logits are then divided by T before softmax.

    T is fitted via L-BFGS over a few hundred steps; in practice it converges
    in <50 iterations.
    """
    name = "temperature_scaling"

    def __init__(self, max_iter: int = 200, lr: float = 0.01, base_metric: str = "macro_f1"):
        self.max_iter = max_iter
        self.lr = lr
        self.base_metric = base_metric
        self.fitted_T: float = 1.0
        self._is_fitted: bool = False

    def score(self, dev_metrics: dict) -> float:
        # Use plain F1 during training; T is fitted only at the end.
        return dev_metrics[self.base_metric]

    @torch.no_grad()
    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            return logits
        return logits / self.fitted_T

    def finalize(
        self,
        model: torch.nn.Module,
        save_dir: Path,
        dev_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> torch.nn.Module:
        if dev_loader is None:
            raise ValueError("TemperatureScalingStrategy.finalize needs dev_loader")

        # Collect dev logits + labels with model frozen.
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_logits.append(outputs["logits"].detach().float().cpu())
                all_labels.append(batch["labels"].detach().cpu())
        logits = torch.cat(all_logits, dim=0)        # [N, 2]
        labels = torch.cat(all_labels, dim=0).long()  # [N]

        # Fit T via L-BFGS minimising NLL = CE(softmax(logits/T), labels).
        T = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        opt = torch.optim.LBFGS([T], lr=self.lr, max_iter=self.max_iter)

        def closure():
            opt.zero_grad()
            scaled = logits.double() / T
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.fitted_T = float(T.detach().clamp_min(1e-3).item())
        self._is_fitted = True

        # Save fitted T alongside the model checkpoint for reproducibility.
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "temperature.json").open("w") as f:
            import json
            json.dump({"fitted_T": self.fitted_T, "name": self.name}, f, indent=2)
        return model

    def export_state(self) -> dict:
        return {"name": self.name, "fitted_T": self.fitted_T, "is_fitted": self._is_fitted}


class SWAStrategy(CheckpointStrategy):
    """Stochastic Weight Averaging over the last K epoch checkpoints
    (Izmailov et al., UAI 2018). During training we snapshot every epoch's
    state-dict; at finalize, we average the LAST K snapshots and load them
    into the model. Score for early stop stays F1.

    Note: we keep up to ``K`` snapshots in CPU memory (so memory footprint is
    K * sizeof(model)). For ~200M-parameter encoders at K=3 this is ~2.4 GB,
    fine on RCP A100-40g.
    """
    name = "swa"

    def __init__(self, k: int = 3, base_metric: str = "macro_f1"):
        if k < 2:
            raise ValueError(f"SWAStrategy needs k>=2 (else just use BestF1), got {k}")
        self.k = int(k)
        self.base_metric = base_metric
        self._snapshots: list[dict] = []
        self._snapshot_epochs: list[int] = []
        self._averaged: bool = False

    def score(self, dev_metrics: dict) -> float:
        return dev_metrics[self.base_metric]

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        dev_metrics: dict,
        dev_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> None:
        # Snapshot the current state dict on CPU and trim to last K.
        snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._snapshots.append(snap)
        self._snapshot_epochs.append(epoch + 1)
        if len(self._snapshots) > self.k:
            self._snapshots = self._snapshots[-self.k:]
            self._snapshot_epochs = self._snapshot_epochs[-self.k:]

    def finalize(
        self,
        model: torch.nn.Module,
        save_dir: Path,
        dev_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> torch.nn.Module:
        if len(self._snapshots) < 2:
            # Training stopped before we had >=2 snapshots; SWA degrades to BestF1.
            return model

        # Average the snapshots, preserving each tensor's original dtype.
        averaged = {}
        for key in self._snapshots[0]:
            ref_dtype = self._snapshots[0][key].dtype
            stacked = torch.stack([s[key].float() for s in self._snapshots], dim=0)
            averaged[key] = stacked.mean(dim=0).to(ref_dtype)
        model.load_state_dict(averaged)
        self._averaged = True

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(averaged, save_dir / "swa_averaged.pt")
        with (save_dir / "swa_meta.json").open("w") as f:
            import json
            json.dump(
                {
                    "name": self.name,
                    "k": self.k,
                    "averaged_epochs": self._snapshot_epochs,
                    "n_snapshots_used": len(self._snapshots),
                },
                f, indent=2,
            )
        return model

    def export_state(self) -> dict:
        return {
            "name": self.name,
            "k": self.k,
            "averaged_epochs": list(self._snapshot_epochs),
            "n_snapshots_used": len(self._snapshots),
            "averaged": self._averaged,
        }


# -------- factory ----------------------------------------------------------


def build_strategy(spec: dict | None) -> CheckpointStrategy:
    """Instantiate a CheckpointStrategy from a YAML-style dict spec.

    spec examples:
        None                            -> BestF1Strategy() (backward-compat default)
        {"name": "best_f1"}             -> BestF1Strategy()
        {"name": "best_f1_positive"}    -> BestF1PositiveStrategy()
        {"name": "composite_f1_ece", "alpha": 1.0} -> CompositeF1ECEStrategy(1.0)
        {"name": "temperature_scaling"} -> TemperatureScalingStrategy()
        {"name": "swa", "k": 3}         -> SWAStrategy(k=3)
    """
    if spec is None:
        return BestF1Strategy()

    if isinstance(spec, str):
        spec = {"name": spec}

    name = spec.get("name", "best_f1")
    if name == "best_f1":
        return BestF1Strategy()
    if name == "best_f1_positive":
        return BestF1PositiveStrategy()
    if name == "composite_f1_ece":
        return CompositeF1ECEStrategy(
            alpha=float(spec.get("alpha", 1.0)),
            base_metric=spec.get("base_metric", "macro_f1"),
        )
    if name == "temperature_scaling":
        return TemperatureScalingStrategy(
            max_iter=int(spec.get("max_iter", 200)),
            lr=float(spec.get("lr", 0.01)),
            base_metric=spec.get("base_metric", "macro_f1"),
        )
    if name == "swa":
        return SWAStrategy(
            k=int(spec.get("k", 3)),
            base_metric=spec.get("base_metric", "macro_f1"),
        )
    raise ValueError(f"unknown checkpoint-strategy name: {name!r}")
