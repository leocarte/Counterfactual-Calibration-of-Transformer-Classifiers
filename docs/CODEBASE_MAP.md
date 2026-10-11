# Codebase map

All paths are relative to the repository root. Experiment settings live in
[configs](../configs/README.md); each entry-point script accepts `--help`.

## Source modules

| Directory | Responsibilities |
|---|---|
| `src/data/` | Text preprocessing, stratified splits, datasets, identity inventory, keyword masking, counterfactual swaps, and heuristic symmetry filtering |
| `src/models/` | Transformer classifier, focal loss, temperature head, LLM prompting, and prompt templates |
| `src/training/` | Baseline, CLP, and CCI trainers; consistency losses; checkpoint strategies; CCI diagnostics |
| `src/calibration/` | Post-hoc multicalibration |
| `src/evaluation/` | Classification and calibration metrics, significance tests, cross-dataset evaluation, robustness, and error analysis |
| `src/utils/` | Configuration loading, logging, and reproducible seeds |

## Entry points

| Script | Purpose |
|---|---|
| `scripts/preprocess_data.py` | Clean GoldStandard2024 and save train/dev/test splits |
| `scripts/train.py` | Train transformer baselines and counterfactual data augmentation configurations |
| `scripts/train_clp.py` | Train counterfactual logit-pairing baselines |
| `scripts/train_cci.py` | Train CCI v2 models and semi-supervised variants |
| `scripts/evaluate.py` | Evaluate trained checkpoints |
| `scripts/apply_multicalibration.py` | Fit and apply post-hoc multicalibration |
| `scripts/run_llm.py` / `scripts/run_llm_pairs.py` | Prompted LLM inference and counterfactual-pair evaluation |
| `scripts/prepare_cross_datasets.py` | Prepare external evaluation datasets |
| `scripts/aggregate_multiseed.py` / `scripts/aggregate_cross_dataset.py` | Aggregate experiment results |
| `scripts/eval_attribution.py` | Compute Integrated Gradients attributions |
| `scripts/make_poster_panel*.py` | Generate the existing poster figures |

## Research artifacts and execution

- [Counterfactual analysis](../analysis/direction1/README.md): numbered analysis
  scripts, a saved report, and statistical results. Cached logits and checkpoints
  are resolved through `SCRATCH_ROOT` or the EPFL scratch paths.
- `analysis/04_attribution_analysis.py`: aggregate attribution outputs.
- `rcp/`: Docker setup, runtime wrappers, and cluster submission scripts. Cluster launchers retain the original EPFL environment settings.
- `results/metrics_from_rcp/`, `results/figures/`, and `results/interpretability/`:
  saved summaries and visualizations.
- `paper/` and `poster/`: LaTeX sources, figures, and existing presentation artifacts.
- `tests/`: automated tests for data handling, swaps, losses, metrics, calibration,
  and training behavior.
- `archive/`: preserved historical documentation and superseded outputs.

Training writes checkpoints, predictions, and metrics to the paths in
`configs/base.yaml`. Raw datasets, local environments, and newly generated training
outputs are excluded from version control.
