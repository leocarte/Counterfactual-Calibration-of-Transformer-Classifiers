# Config taxonomy (`configs/`)

51 YAML configs. Files are kept flat (not in sub-folders) because every `rcp/submit_*.sh` and `scripts/train*.py` references them by literal path, and breaking those references would be risky this close to submission. This README is the navigation aid instead.

**Naming convention:**

```
{family}[_{variant}][_{architecture}][_smoke].yaml
```

`smoke` = 20-tweet subset for validating that the job pipeline (RCP submission, image, mounts, env vars) works before burning hours on the full 1,060-tweet test set.

---

## Shared defaults

| Config | What it is |
|---|---|
| `base.yaml` | Shared defaults (seeds, paths, batch size, lr, scheduler) inherited by every other config |

## Fine-tuned baselines

| Config | Backbone | Loss | Notes |
|---|---|---|---|
| `roberta.yaml` | RoBERTa-base | Weighted CE | **Model A**, baseline transformer + class weighting |
| `deberta.yaml` | DeBERTa-v3-base | Weighted CE | **Model B**, baseline with disentangled attention |
| `hatebert.yaml` | HateBERT | Weighted CE | Hate-specialised pretraining baseline |
| `deberta_focal_mask.yaml` | DeBERTa-v3-base | Focal + identity-token mask | **Model C, the strong baseline** |
| `model_c_deberta_v3_large.yaml` | DeBERTa-v3-large | Focal + identity-token mask | Model C scaled up |
| `model_c_roberta_large.yaml` | RoBERTa-large | Focal + identity-token mask | Model C, different family |

## CCI v2 (our method)

| Config | T | Divergence | Notes |
|---|---|---|---|
| **`cci_v2_fixed_js.yaml`** | fixed at 1.0 | Jensen-Shannon | **THE PIVOT**, headline numbers come from this |
| `cci_v2_learned_js.yaml` | per-group learnable | JS | Ablation cell (originally the pre-registered pivot) |
| `cci_v2_fixed_l1.yaml` | fixed | L1 | Ablation (lowest CFR but worst FNED) |
| `cci_v2_learned_l1.yaml` | learnable | L1 | Ablation |
| `cci_v2_fixed_js_deberta_v3_large.yaml` | fixed | JS | Pivot on DeBERTa-v3-large |
| `cci_v2_fixed_js_roberta_large.yaml` | fixed | JS | Pivot on RoBERTa-large (cross-family) |

## Davani CLP baselines

| Config | λ | Filter | Notes |
|---|---|---|---|
| `davani_lambda05.yaml` | 0.5 | heuristic | **The strongest CLP comparator**, referenced as "Davani λ=0.5" in the paper |
| `davani_lambda10.yaml` | 1.0 | heuristic | |
| `davani_lambda20.yaml` | 2.0 | heuristic | |
| `davani_lambda10_nofilt.yaml` | 1.0 | none | Isolates the symmetry filter's contribution |

## Calibration

| Config | Strategy |
|---|---|
| `calibration_F1.yaml` | F1-based early stopping (the bimodal-ECE setting) |
| `calibration_SWA.yaml` | Stochastic Weight Averaging |
| `calibration_TS.yaml` | Post-hoc Temperature Scaling, Lemma 1 verification |
| `calibration_composite_a05.yaml` | Composite F1 − 0.05·ECE checkpoint score |
| `calibration_composite_a10.yaml` | Composite F1 − 0.10·ECE |
| `calibration_composite_a20.yaml` | Composite F1 − 0.20·ECE |

## Prompted LLMs

Three prompt strategies × three local LLMs = 9 configs. `D1 = zero-shot, D2 = few-shot (5+5), D3 = Guided Chain-of-Thought (Patel EMNLP 2025)`.

| LLM | D1 | D2 | D3 |
|---|---|---|---|
| Llama-3.1-8B-Instruct | `llm_zeroshot.yaml` | `llm_fewshot.yaml` | `llm_guidedcot.yaml` |
| Mistral-7B-Instruct | `mistral_zeroshot.yaml` | `mistral_fewshot.yaml` | `mistral_guidedcot.yaml` |
| Qwen-2.5-7B-Instruct | `qwen_zeroshot.yaml` | `qwen_fewshot.yaml` | `qwen_guidedcot.yaml` |


## Smoke variants

Most families ship a `*_smoke.yaml` (20-tweet, 1-epoch) to validate the job pipeline before a full run, via `bash rcp/submit_*.sh smoke`. The per-architecture and LLM configs have one each; CCI v2 and Davani share a single generic smoke (`cci_v2_smoke.yaml`, `davani_smoke.yaml`); the calibration configs have none.

## Adding a new config

1. Decide which family it belongs to.
2. Inherit from `base.yaml` via `defaults: [base]`.
3. Add a one-line description as the first comment of the file.
4. If you want it on RCP, add a launcher in `rcp/submit_<family>.sh`.
5. Add an entry to the appropriate table above.
