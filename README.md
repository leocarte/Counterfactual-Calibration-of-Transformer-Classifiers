# Antisemitism Detection in Social Media Text

**EE-559 Deep Learning Mini-Project — EPFL, Spring 2026**

> "Where Criticism Ends and Hate Begins: Fine-Tuned Transformers and a Counterfactual
> Calibration-Invariance Loss for Antisemitism Detection in Social Media Text"

## Overview

Antisemitism detection with a **three-axis evaluation framework** — positive-class F1,
group-conditional ECE per identity keyword, and counterfactual stability under
identity-token swaps within a religiously-symmetric vocabulary — on the GoldStandard2024
dataset (Jikeli et al., 2024, 11,311 IHRA-annotated tweets).

**Headline contribution: Counterfactual Calibration Invariance (CCI v2).** A training-time
loss penalising the Jensen-Shannon divergence between predictions on a tweet and its
identity-swapped counterfactual. CCI v2 reduces the hard counterfactual flip rate by
**−49 % on DeBERTa-v3-base, −55 % on DeBERTa-v3-large, and −73 % on RoBERTa-large**
(cross-family), with positive-class F1 preserved or improved at scale (0.72 best).

**Headline finding: multicalibration vs counterfactual-invariance tension.** Post-hoc
multicalibration cuts hard CFR by 8 % but *inflates* L1-norm soft CFR by 96 %. To our
reading the first quantitative report of this trade-off on a text classifier.

## Quick start

```bash
# Clone and setup
git clone https://github.com/giacomocase/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt

# Download GoldStandard2024 (CC BY 4.0) — manual download from Zenodo
# https://zenodo.org/records/14448399
# Place at data/raw/GoldStandard2024.csv, then run:
python scripts/preprocess_data.py

# Local training (single GPU)
python scripts/train.py     --config configs/deberta_focal_mask.yaml   # Model C
python scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml      # CCI v2 pivot

# Full evaluation (after training)
python scripts/evaluate.py \
  --config configs/cci_v2_fixed_js.yaml \
  --checkpoint results/checkpoints/cci_v2_fixed_js/seed42/best_model.pt

# RCP cluster (EPFL): see docs/onboarding/RCP_QUICK_REFERENCE.md
bash rcp/submit.sh <job_name> "python3 scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml"
```

## Repository structure

```
DeepLearning/                   ← repo root (single project, flat layout)
├── README.md                   ← you are here
├── LICENSE                     MIT
├── pyproject.toml / setup.cfg / requirements.txt
│
├── src/                        Python source (data, models, training, evaluation, calibration, utils)
├── scripts/                    Entry-point scripts (train, evaluate, run_llm, prepare_cross_datasets, ...)
├── tests/                      pytest unit tests (~50 tests, 16 files)
│
├── configs/                    51 YAML configs — see configs/README.md for taxonomy
├── rcp/                        Run:ai cluster scaffold (Dockerfile, submit.sh, _runner.sh)
│
├── analysis/                   paper-driving analysis pipeline
│   ├── direction1/             nine scripts that produce REPORT.md
│   └── 04_attribution_analysis.py   Leonardo's IG aggregation
│
├── paper/                      LaTeX sources (paper/sections/*.tex); main.pdf is gitignored
│   ├── main.tex, references.bib
│   ├── sections/               nine .tex files
│   ├── figures/                attribution_shift.pdf
│   └── notes/                  internal team notes (gitignored)
│
├── docs/                       team documentation (onboarding, decisions, plans)
│   ├── onboarding/             new-member briefs + LLM prompt template + codebase map
│   ├── decisions/              DECISIONS_LOG.md + KEY_CITATIONS.md
│   └── team/                   team plan + bring-up debugging history
│
├── data/                       gitignored (license-restricted raw data)
└── results/                    gitignored (checkpoints, plots, per-seed CSVs — live on RCP scratch)
```

## Documentation entry points

- **New team member?** Read `docs/onboarding/PROJECT_OVERVIEW.md` first (3 min), then your
  vertical brief (`LEONARDO_BRIEF.md` or `YASMIN_BRIEF.md`).
- **Coming up to speed on the codebase?** `docs/onboarding/CODEBASE_MAP.md`.
- **Running on the cluster?** `docs/onboarding/RCP_QUICK_REFERENCE.md`.
- **Configuration files?** `configs/README.md` (51 configs, one-table taxonomy).
- **The analysis pipeline?** `analysis/direction1/README.md`.
- **Compiling the paper?** `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`. PDF stays local (excluded by `.gitignore`).

## Dataset

**GoldStandard2024** (Jikeli et al., 2024) — 11,311 English tweets annotated for
antisemitism under the IHRA Working Definition. Test split: 1,060 examples (187 positives).
Keyword stratification: Jews 41.4 %, Israel 41.4 %, Kikes 8.6 %, ZioNazi 8.6 %.
CC BY 4.0 license. Download: https://zenodo.org/records/14448399

## Results snapshot (numbers from `analysis/direction1/REPORT.md`)

| Method | F1⁺ | ECE | CFR_hard | FNED |
|---|---:|---:|---:|---:|
| Model A — RoBERTa-base | 0.61 | 0.10 | 0.060 | 0.66 |
| Model B — DeBERTa-v3 vanilla | 0.65 | 0.06 | 0.044 | 0.58 |
| Model C — DeBERTa + focal + kw-mask | 0.66 | 0.07 | 0.036 | 0.58 |
| Davani CLP λ=0.5 | 0.66 | 0.07 | 0.040 | **0.53** |
| **CCI v2 (fixed T, JS) — pivot** | **0.68** | 0.07 | **0.018** | 0.54 |
| CCI v2 on DeBERTa-v3-large | **0.72** | 0.06 | 0.019 | 0.43 |
| CCI v2 on RoBERTa-large (cross-family) | 0.69 | 0.08 | **0.013** | 0.56 |

All CCI v2 CFR improvements survive Holm-Bonferroni in a pre-specified two-family suite
(23/54 main + 6/6 within-architecture transfer). See `paper/sections/results.tex` §4.4–4.5
for the full breakdown.

## Citation

```bibtex
@misc{case2026antisemitism,
  title={Where Criticism Ends and Hate Begins: Fine-Tuned Transformers and a
         Counterfactual Calibration-Invariance Loss for Antisemitism Detection},
  author={Case, Giacomo and Chaqor, Yasmin and Cartesegna, Leonardo},
  year={2026},
  institution={EPFL EE-559 Deep Learning}
}
```

## License

Code: MIT. Dataset (GoldStandard2024): CC BY 4.0.
