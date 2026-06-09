# Antisemitism Detection in Social Media Text

EE-559 Deep Learning mini-project, EPFL, Spring 2026.

> "Where Criticism Ends and Hate Begins: Fine-Tuned Transformers and a Counterfactual
> Calibration-Invariance Loss for Antisemitism Detection in Social Media Text"

## Overview

We detect antisemitism in tweets (GoldStandard2024; Jikeli et al., 2024) and evaluate
predictions along three axes: positive-class F1, group-conditional calibration error
(ECE), and counterfactual stability under identity-token swaps.

The headline contribution is Counterfactual Calibration Invariance (CCI v2), a
training-time loss penalising the Jensen-Shannon divergence between a tweet's prediction
and its identity-swapped counterfactual. It cuts the hard counterfactual flip rate by
49% / 55% / 73% on DeBERTa-v3-base, DeBERTa-v3-large, and RoBERTa-large, with F1 preserved
or improved (0.72 best). We also report a tension between multicalibration and
counterfactual invariance: post-hoc multicalibration lowers hard CFR by 8% but inflates
soft CFR by 96%.

## Quick start

```bash
# Setup
pip install -r requirements.txt

# The dataset (GoldStandard2024, CC BY 4.0) ships under data/raw/.
# To refetch it from Zenodo instead, run: bash data/download.sh
python scripts/preprocess_data.py

# Local training (single GPU)
python scripts/train.py     --config configs/deberta_focal_mask.yaml   # Model C baseline
python scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml      # CCI v2 pivot

# Evaluation (after training)
python scripts/evaluate.py \
  --config configs/cci_v2_fixed_js.yaml \
  --checkpoint results/checkpoints/cci_v2_fixed_js/seed42/best_model.pt

# Run:ai cluster (EPFL RCP): the rcp/ directory holds the submit scaffold
bash rcp/submit.sh <job_name> "python3 scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml"
```

## Repository layout

```
.
├── src/                                importable library
│   ├── data/
│   │   ├── counterfactual_swap.py        identity-token swap engine (feeds CCI / CLP)
│   │   ├── symmetry_classifier.py        heuristic filter for label-preserving swaps
│   │   ├── identity_inventory.py         religiously-symmetric identity vocabulary
│   │   ├── preprocessing.py              tweet cleaning + keyword masking
│   │   ├── dataset.py / cf_dataset.py    torch datasets (plain / counterfactual-paired)
│   │   ├── splits.py                     stratified train/dev/test splits
│   │   └── keyword_masking.py            keyword-reliance flip-rate check
│   ├── models/
│   │   ├── transformer_classifier.py     HF encoder + classification head
│   │   ├── temperature_head.py           per-group temperature head (CCI v2)
│   │   ├── focal_loss.py                 focal loss (Model C)
│   │   └── llm_prompter.py + prompts/    zero-/few-shot / guided-CoT LLM baselines
│   ├── training/
│   │   ├── trainer.py                    base fine-tuning loop
│   │   ├── cci_loss.py / cci_trainer.py  CCI v2 contribution (JS pair-divergence)
│   │   ├── clp_loss.py / clp_trainer.py  Davani CLP baseline
│   │   ├── checkpoint_strategies.py      calibration bake-off strategies
│   │   └── cci_diagnostics.py            temperature-shortcut kill-criterion
│   ├── evaluation/                       metrics, significance (paired bootstrap),
│   │                                     cross_dataset, robustness, error_analysis
│   ├── calibration/                      post-hoc multicalibration
│   └── utils/                            config (OmegaConf), seeding, logging
├── scripts/                            CLI entry points
│   ├── preprocess_data.py                build splits from the raw CSV
│   ├── train.py / train_cci.py / train_clp.py    baselines / CCI v2 / Davani
│   ├── run_llm.py                        LLM inference (Llama / Mistral / Qwen)
│   ├── evaluate.py / apply_multicalibration.py   full eval + post-hoc calibration
│   ├── aggregate_*.py / threshold_sweep_cross_dataset.py / prepare_cross_datasets.py
│   └── eval_attribution.py                per-checkpoint integrated-gradients
├── analysis/                           checkpoints -> paper numbers and figures (flat)
│   ├── 01_…_09_*.py                      CFR pipeline: manifest, forward pass, CFR/FPED-FNED/ECE, significance
│   ├── _utils.py                         shared pipeline helpers
│   ├── REPORT.md                         generated results report (CFR / ECE / FPED-FNED / significance numbers)
│   ├── PIPELINE.md                       pipeline documentation
│   └── attribution_analysis.py           integrated-gradients interpretability
├── configs/                            51 YAML experiment configs 
├── rcp/                                EPFL Run:ai cluster scaffold
│   ├── Dockerfile / requirements.txt / build_and_push.sh
│   ├── submit.sh / _runner.sh             core submit wrapper + container entry point
│   └── submit_*.sh                        per-experiment launchers (cci_v2, davani, llm, ...)
├── tests/                              16 pytest files (losses, swaps, metrics, calibration, ...)
├── data/
│   ├── raw/GoldStandard2024.csv          dataset (CC BY 4.0)
│   ├── external/                         HateXplain / ToxiGen cross-dataset subsets
│   └── download.sh                       refetch from Zenodo
└── deliverables/                         final submission deliverables
    ├── poster.pdf                        conference-style poster
    ├── report.pdf                        3-page paper (compiled PDF)
    └── screencast.mp4                    project video / screencast
```

Root files: `README.md`, `LICENSE` (MIT), `requirements.txt`, `pyproject.toml`, `setup.cfg`.

The `deliverables/` directory collects the final deliverables for submission: the poster
(`poster.pdf`), the paper (`report.pdf`), and the project screencast.

## Dataset

GoldStandard2024 (Jikeli et al., 2024): 11,311 English tweets annotated for antisemitism
under the IHRA Working Definition. Test split: 1,060 examples, 187 positive. Keyword
stratification: Jews 41.4%, Israel 41.4%, Kikes 8.6%, ZioNazi 8.6%. Licensed CC BY 4.0.
Source: https://zenodo.org/records/14448399

## License

Code: MIT. Dataset (GoldStandard2024): CC BY 4.0.
