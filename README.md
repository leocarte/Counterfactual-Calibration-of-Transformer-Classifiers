# Counterfactual Calibration of Transformer Classifiers

**EE-559 Deep Learning Mini-Project, EPFL, Spring 2026**

This repository contains the final code, analysis, poster, and report for a deep
learning project on robust antisemitism detection in social-media text. The main
contribution is **Counterfactual Calibration Invariance (CCI v2)**, a PyTorch
training objective that penalizes Jensen-Shannon divergence between predictions
on an input text and label-preserving identity-swapped counterfactuals.

## Highlights

- Fine-tuned DeBERTa-v3 and RoBERTa transformer classifiers with focal loss for
  imbalanced binary text classification.
- Introduced CCI v2 as a training-time regularizer for counterfactual prediction
  stability.
- Compared against focal-loss baselines, keyword masking, Davani-style CLP,
  temperature scaling, multicalibration, and prompted 8B LLM baselines.
- Reduced hard counterfactual flip rates by 49% to 73% across DeBERTa and
  RoBERTa backbones while preserving strong positive-class F1.
- Evaluated calibration, counterfactual robustness, subgroup error metrics,
  cross-dataset transfer, and Integrated Gradients attribution behavior.

## Final Artifacts

- [Final report](report.pdf)
- [Final poster](poster.pdf)
- [Analysis report](analysis/direction1/REPORT.md)

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── poster.pdf
├── report.pdf
│
├── src/                    Core Python package
│   ├── data/               Preprocessing, splits, identity swaps, datasets
│   ├── models/             Transformer classifiers, focal loss, LLM prompting
│   ├── training/           Baseline, CLP, and CCI trainers/losses
│   ├── calibration/        Multicalibration utilities
│   ├── evaluation/         Metrics, robustness, cross-dataset evaluation
│   └── utils/              Config, logging, seed helpers
│
├── scripts/                Training, evaluation, aggregation, plotting scripts
├── configs/                YAML experiment configurations
├── tests/                  Unit tests for data, loss, training, and metrics code
├── analysis/               Final analysis pipeline and generated summary report
├── results/                Final figures and interpretability summaries
├── paper/                  LaTeX source for the project report
└── data/                   Dataset download helper only; raw data is excluded
```

## Setup

```bash
pip install -r requirements.txt
```

The raw datasets are not included in this public-facing copy. Use
`data/download.sh` and the dataset instructions in the report to recreate the
expected local data layout.

## Example Commands

```bash
# Preprocess data after placing the raw dataset locally
python scripts/preprocess_data.py

# Train the focal-loss baseline
python scripts/train.py --config configs/deberta_focal_mask.yaml

# Train the CCI v2 model
python scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml

# Run evaluation from a trained checkpoint
python scripts/evaluate.py \
  --config configs/cci_v2_fixed_js.yaml \
  --checkpoint results/checkpoints/cci_v2_fixed_js/seed42/best_model.pt
```

## Results Snapshot

| Method | F1+ | ECE | CFR hard | FNED |
|---|---:|---:|---:|---:|
| Model A, RoBERTa-base | 0.61 | 0.10 | 0.060 | 0.66 |
| Model B, DeBERTa-v3 vanilla | 0.65 | 0.06 | 0.044 | 0.58 |
| Model C, DeBERTa + focal + keyword masking | 0.66 | 0.07 | 0.036 | 0.58 |
| Davani CLP, lambda=0.5 | 0.66 | 0.07 | 0.040 | **0.53** |
| **CCI v2, fixed T + JS** | **0.68** | 0.07 | **0.018** | 0.54 |
| CCI v2 on DeBERTa-v3-large | **0.72** | 0.06 | 0.019 | 0.43 |
| CCI v2 on RoBERTa-large | 0.69 | 0.08 | **0.013** | 0.56 |

For the full breakdown, see [analysis/direction1/REPORT.md](analysis/direction1/REPORT.md)
and the final [report](report.pdf).

## License

Code is released under the MIT License. Dataset files are not redistributed in
this repository.
