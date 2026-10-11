# Counterfactual Robustness and the Calibration Trap in Antisemitism Detection

**EE-559 Deep Learning Project · EPFL · Spring 2026**

A research project investigating whether text classifiers can detect antisemitism
without relying on identity terms as shortcuts. We evaluate RoBERTa and DeBERTa
models on classification quality, confidence calibration, and prediction stability
under identity-token swaps.

The main method, **Counterfactual Calibration Invariance (CCI v2)**, adds a
Jensen–Shannon consistency loss between an original text and its counterfactual.
A heuristic symmetry filter excludes swaps involving slurs or geopolitical-only
references. The repository includes training, calibration, statistical analysis,
and Integrated Gradients attribution workflows.

## Results

Across six seeds, CCI v2 reduces the hard counterfactual flip rate by **48.9%** on
DeBERTa-v3-base while improving positive-class F1 from **0.6597 to 0.6767**.
The reduction also holds on DeBERTa-v3-large (**55.2%**) and RoBERTa-large
(**72.9%**), relative to the same-architecture focal-loss and keyword-masking baseline.

| Backbone | Method | Positive-class F1 ↑ | ECE ↓ | Hard CFR ↓ |
|---|---|---:|---:|---:|
| DeBERTa-v3-base | Focal loss + keyword masking (Model C) | 0.6597 | 0.0772 | 0.0358 |
| DeBERTa-v3-base | **CCI v2 (fixed T, JS)** | **0.6767** | 0.0981 | **0.0183** |
| DeBERTa-v3-large | Focal loss + keyword masking | 0.7041 | 0.0966 | 0.0417 |
| DeBERTa-v3-large | **CCI v2 (fixed T, JS)** | **0.7194** | 0.0986 | **0.0187** |
| RoBERTa-large | Focal loss + keyword masking | 0.6894 | 0.0678 | 0.0465 |
| RoBERTa-large | **CCI v2 (fixed T, JS)** | **0.6935** | 0.0766 | **0.0126** |

Values are means from the saved [analysis report](analysis/direction1/REPORT.md),
which includes standard deviations, group-level error rates, and significance tests.
Percentage reductions above are calculated from the displayed means. ECE is expected
calibration error; hard CFR is the fraction of counterfactual pairs whose predicted
class changes at a 0.5 threshold, evaluated on a fixed 2,117-pair test manifest.

The experiments also expose a trade-off: post-hoc multicalibration lowers Model C's
ECE from 0.0772 to 0.0354, but raises soft CFR (L1) from 0.0322 to 0.0630.
CCI improves counterfactual stability without consistently improving calibration
or every group fairness metric. See the [paper (PDF)](paper/woah2026/woah2026_short.pdf) for methodology
and limitations.

## Getting started

Run the following commands from the repository root. Python 3.10+ is required;
a CUDA-capable GPU is recommended for training. Model and tokenizer weights are
downloaded from Hugging Face on first use.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Windows, activate the environment with `.venv\Scripts\activate`.
The core installation covers transformer training, evaluation, and tests.
Optional LLM inference and notebooks use additional dependencies:

```bash
# Quantized LLM inference; requires compatible GPU hardware
python -m pip install "bitsandbytes>=0.42.0"

# Cluster LLM jobs install vLLM at runtime; see rcp/submit_llm.sh
# Optional notebooks
python -m pip install "jupyterlab>=4.0.0" "ipywidgets>=8.1.0"
```

### Prepare the dataset

Download GoldStandard2024 from the
[dataset record](https://zenodo.org/records/14448399) and save the CSV as
`data/raw/GoldStandard2024.csv`. Alternatively, use the existing download script
with `wget` installed:

```bash
bash data/download.sh
python scripts/preprocess_data.py
```

The preprocessing pipeline cleans text, removes duplicate texts, and produces
label- and keyword-stratified 80/10/10 train/dev/test splits. The reported test
split contains 1,060 examples, including 187 positives.

### Train and evaluate

```bash
# Strong baseline: DeBERTa-v3-base + focal loss + keyword masking
python scripts/train.py --config configs/deberta_focal_mask.yaml --seed 42 --no-wandb

# Main method: CCI v2 with fixed temperature and JS divergence
python scripts/train_cci.py --config configs/cci_v2_fixed_js.yaml --seed 42 --no-wandb

# Evaluate the checkpoint created by the command above
python scripts/evaluate.py \
  --config configs/cci_v2_fixed_js.yaml \
  --checkpoint results/checkpoints/cci_v2_fixed_js_seed42/best_model.pt
```

`--no-wandb` disables experiment tracking. Omit it to use Weights & Biases.
Explicit seeds are appended to experiment names; checkpoints are saved under
`results/checkpoints/<experiment>_seed<seed>/`. Repeat training with seeds
42–47 to match the reported six-seed setup. The full counterfactual and statistical
analysis follows the [analysis pipeline](analysis/direction1/README.md) and requires
the trained checkpoints and cached predictions. Large training artifacts are
stored outside the published repository; saved summaries and figures are included.

### Run tests

```bash
python -m pytest -q
```

The tests cover counterfactual swaps, symmetry filtering, loss functions,
calibration, metrics, dataset handling, and checkpoint behavior. Tokenizer-based
tests download `bert-base-uncased` on first use.

## Repository guide

| Path | Contents |
|---|---|
| [src/](src/) | Data processing, models, training, calibration, evaluation, and utilities |
| [scripts/](scripts/) | Training, evaluation, LLM inference, attribution, and plotting entry points |
| [configs/](configs/README.md) | YAML experiment configurations and ablations |
| [tests/](tests/) | Automated tests |
| [analysis/](analysis/direction1/README.md) | Counterfactual analysis, significance testing, and saved report |
| [results/](results/) | Saved experiment summaries and figures |
| [paper/](paper/) | [Research paper (PDF)](paper/woah2026/woah2026_short.pdf), LaTeX sources, and figures |
| [poster/](poster/) | Research poster and method diagrams |
| [rcp/](rcp/) | EPFL cluster submission and runtime scripts |
| [docs/](docs/README.md) | Codebase guide and research workflow documentation |
| [archive/](archive/README.md) | Historical plotting and cluster helper scripts |

## Authors and citation

Giacomo Case, Yasmin Chaqor, and Leonardo Cartesegna — EPFL EE-559 Deep Learning.

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

Code is released under the [MIT License](LICENSE). GoldStandard2024 is distributed
separately under CC BY 4.0; consult its dataset record for attribution. Results are
specific to English Twitter/X text and the dataset's IHRA-based annotations.
