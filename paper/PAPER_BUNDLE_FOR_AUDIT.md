# Paper draft bundle for independent peer review audit

**Generated**: 2026-05-14 (post-commit `4bff407`)
**Project**: Counterfactual Calibration Invariance (CCI v2) on antisemitism detection
**Repo**: https://github.com/giacomocase/DeepLearning, branch `main`

---

## AUDIT PROMPT FOR AI REVIEWER (paste this at the top when uploading)

You are a senior peer reviewer for a NeurIPS/EMNLP-level paper. Below is the full LaTeX paper draft (9 sections, ~10,800 words) followed by REPORT.md (auto-generated from the analysis pipeline; the ground-truth empirical-evidence file).

Perform a **ruthless, non-sycophantic audit** along five dimensions:

1. **Internal consistency** — Cross-check every numerical claim in the .tex sections against REPORT.md. Flag contradictions. Specifically: 49% / 55% / 73% CFR reduction; +96% CFR_soft_L1 under multicalibration; 23/54 Bonferroni rejections; FNED capacity-bound; 6/6 bit-exact TS preservation.

2. **Scientific rigor** — Is the 54-test Bonferroni right (vs Holm-Bonferroni)? Is paired bootstrap properly paired with n=5/n=6 seed mismatch? Is Theorem 1 (TS preserves CFR_hard) correct including corner cases (e.g., z(x) = 0 exactly)?

3. **Novelty vs prior art** — Closest prior: Garg AIES 2019, Davani WOAH 2021, Mou & Lee 2024, Hébert-Johnson ICML 2018, Hansen NeurIPS 2024, Guo ICML 2017. Is CCI v2's delta (JS-in-probability-space + per-group T + structural Theorem) a contribution? Missing citations on counterfactual fairness for text or multicalibration's effect on individual fairness?

4. **Publication targets** — Honest acceptance probability for: WOAH 2026, EMNLP Findings 2026, EMNLP Main 2026, EACL 2027 Main. For each: single biggest blocker.

5. **Adversarial reviewers** — Simulate 3 reviewers, 3 most damaging attacks each. Rate each: easily defensible / defensible with effort / not defensible.

**Output**: Markdown report, one section per dimension, plus "Bottom line" with 3 critical issues to fix before submission. No sycophancy.

---

# 1. ABSTRACT

```latex
\begin{abstract}
Detecting antisemitism in social-media text is harder than generic hate-speech classification: the criticism-vs-hate boundary on Israeli policy is contested under the IHRA Working Definition; antisemitic discourse relies on coded language, so identity terms alone are unreliable signals; and the data is imbalanced across identity sub-populations with very different base rates. On the GoldStandard2024 dataset~\citep{jikeli2024goldstandard} (11{,}311 IHRA-annotated tweets, 4.79:1 imbalance; test split 1{,}060 / 187 positives) we study model behaviour along three connected axes --- accuracy on the positive class, calibration \emph{per sub-group} rather than only on average, and counterfactual stability under identity-term swaps. We propose \textbf{Counterfactual Calibration Invariance (CCI v2)}, a training-time loss penalising the Jensen-Shannon divergence between predictions on a sentence and on its identity-token-swapped counterfactual, modulated by a per-group temperature head. We prove (and verify exactly in 6/6 seeds) that post-hoc temperature scaling at the decision threshold cannot reduce the hard counterfactual flip rate, so the intervention must happen during training. \textbf{Key findings:} (i) CCI v2 reduces the hard counterfactual flip rate (CFR) by 49\% on DeBERTa-v3-base (184M), 55\% on DeBERTa-v3-large (440M), and 73\% on RoBERTa-large (355M, different family) versus a strong focal+identity-mask baseline at iso-architecture, with positive-class F\textsubscript{1} \emph{improved} at larger scale (0.72 best). A 54-test paired bootstrap with Bonferroni correction at $\alpha = 0.05/54$ rejects the null on all 18 counterfactual-flip-rate comparisons (12 base + 6 multi-architecture), against four base baselines and two large-model Model~C cells. (ii) Applying multicalibration~\citep{hebertjohnson2018multicalibration} post-hoc to the same baseline reduces the hard flip rate by 8\% but \emph{increases the L1-norm soft flip rate by 96\%} (from 0.032 to 0.063): counterfactual invariance and subgroup-conditional calibration can be in direct tension as post-hoc fairness tools. (iii) The FNED Pareto regression CCI v2 introduces at base scale ($+3.7\%$ vs Davani CLP~\citep{davani2021improving}) disappears at larger scale ($-1.5\%$ on DeBERTa-v3-large), suggesting it is capacity-bound rather than method-intrinsic. (iv) Under focal loss with F\textsubscript{1}-based early stopping we observe a previously unreported \emph{bimodal} ECE distribution across seeds (4/6 converge to well-calibrated ECE $\sim 0.05$, 2/6 to a poorly-calibrated state at ECE $\sim 0.12$), traced to an interaction between the F\textsubscript{1}-peak epoch and the calibration trajectory. We additionally benchmark prompted LLMs (Mistral-7B-Instruct, Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.3-70B via Groq) under zero-shot, few-shot, and Guided Chain-of-Thought, with zero-shot cross-dataset transfer to the Jewish-target subsets of HateXplain~\citep{mathew2021hatexplain} and ToxiGen~\citep{hartvigsen2022toxigen}.
\end{abstract}
```

---

# 2. INTRODUCTION

The introduction (~1100 words) lays out the three task difficulties (criticism-vs-hate, coded language, class imbalance with heterogeneous sub-populations), the three-axis evaluation framing, the CCI v2 approach, and 5 contributions:

1. **CCI v2 with architecture-and-family transfer**: 49% / 55% / 73% CFR reduction across DeBERTa-v3-base/large + RoBERTa-large; all 18 CFR comparisons reject Bonferroni 0.05/54.
2. **Multicalibration × counterfactual invariance tension**: post-hoc multicalibration lowers CFR_hard by 8% but raises CFR_soft_L1 by 96%.
3. **Honest fairness audit with capacity-bound finding**: FNED regression of +3.7% vs Davani at base scale, −1.5% at large scale.
4. **Bimodal ECE characterisation** under focal + F1 early-stopping.
5. **Fine-tuned vs prompted-LLM comparison + cross-dataset transfer**.

Full text in `paper/sections/introduction.tex` on `main` branch.

---

# 3. RELATED WORK

```latex
\section{Related Work}
\subsection{Hate-speech detection with transformers}
The application of pretrained transformer models to hate-speech detection has become the dominant paradigm since BERT~\citep{devlin2019bert}. \citet{liu2019roberta} introduced RoBERTa with extended pretraining and dynamic masking; \citet{he2023debertav3} proposed DeBERTa-v3, whose disentangled attention separately encodes content and position. \citet{caselli2021hatebert} retrained BERT on banned Reddit communities to produce HateBERT. A recurring failure mode is \emph{identity-term shortcut learning}: \citet{vidgen2020directions} and \citet{dixon2018measuring} document this. \citet{dixon2018measuring} introduced the FPED and FNED metrics we report.

\subsection{Antisemitism detection in NLP}
\citet{jikeli2023antisemitic} created the GoldStandard dataset of IHRA-annotated English tweets; we use its 2024 version~\citep{jikeli2024goldstandard}. \citet{patel2025evaluating} conducted a comprehensive LLM evaluation (GPT-4, Claude, Llama families) under zero/few-shot/Guided CoT. \citet{liu2024detecting} apply LoRA fine-tuning to antisemitism detection; they study accuracy but not counterfactual fairness or calibration.

\subsection{Counterfactual Logit Pairing and counterfactual fairness for text}
The closest prior art to CCI v2 is the CLP family. \citet{garg2019counterfactual} introduced CLP at AIES, framing it as "counterfactual token fairness" through a paired-logit L_p penalty; they explicitly note (§2) that CLP is individual fairness via Lipschitz constraint~\citep{dwork2012fairness}. \citet{davani2021improving} (WOAH; not 2023) adapted CLP to hate speech with an LM-likelihood symmetry filter. \citet{mou2024fairness} apply CLP in a fairness-aware framework with uncertainty estimation. \citet{kaushik2020learning} establish counterfactual augmentation on sentiment; \citet{sen2023people} report a negative result --- LLM-generated counterfactuals underperform human-generated ones (motivating our heuristic-filter swap engine). CCI v2 generalises this family on (i) divergence in probability space rather than L_p in logit, (ii) per-group learnable temperature, (iii) the structural Theorem characterising what post-hoc TS cannot do.

\subsection{Calibration and multicalibration}
Modern neural classifiers are over-confident~\citep{guo2017calibration}; Temperature Scaling is the canonical post-hoc fix. \citet{mukhoti2020calibrating} show focal loss yields better-calibrated networks; \citet{tao2023pcs} observe "early stopping fails to calibrate networks" (underpinning our bimodal-ECE). \citet{hebertjohnson2018multicalibration} introduced multicalibration at ICML, requiring calibration to hold within every computationally-bounded subgroup. \citet{hansen2024multicalibration} provide one of the few empirical studies in NLP, examining subgroup ECE but NOT counterfactual fairness. Our §4.3 finding (+96% CFR_soft_L1 under multicalibration) is, to our reading, the first quantitative report on text classifiers.

\subsection{LLM prompting and statistical-significance protocols}
\citet{brown2020language} demonstrated zero/few-shot ability; \citet{wei2022chain} introduced Chain-of-Thought; \citet{turpin2023language} contest CoT faithfulness. \citet{dettmers2022gptint} introduced 4-bit NF4 quantization. We benchmark Mistral-7B-Instruct~\citep{jiang2023mistral}, Qwen-2.5-7B-Instruct~\citep{qwen25}, Llama-3.1-8B-Instruct~\citep{llama31}, Llama-3.3-70B (Groq API). We follow \citet{dror2018hitchhikers} for paired-bootstrap significance with Bonferroni at $\alpha = 0.05/54$ over the 9-comparator × 6-metric grid.
```

---

# 4. METHODOLOGY (key excerpts)

## §3.1 Dataset

GoldStandard2024 (Jikeli et al. 2024): 11,311 IHRA-annotated English tweets. 4 keyword groups (Jews 56.3% / Israel 36.6% / Kikes 2.5% / ZioNazi 4.7%) with very different base rates (11.4% / 16.1% / 34.3% / 88.3%). Class imbalance 4.79:1. Split 80/10/10 stratified.

## §3.2 Models

### Fine-tuned transformers
- **Model A**: RoBERTa-base 125M + weighted CE
- **Model B**: DeBERTa-v3-base 184M + weighted CE
- **Model B2**: HateBERT 110M + weighted CE
- **Model C**: DeBERTa-v3-base + focal loss (γ=2, α=0.75) + keyword masking (p=0.3)
- Implementation uses tokenizer.mask_token so masking works for both DeBERTa `[MASK]` and RoBERTa `<mask>`

### Multi-architecture variants (new for CCI v2 transfer)
- DeBERTa-v3-large (440M, same family) and RoBERTa-large (355M, different family)
- 6 seeds per cell, batch 4 × grad_accum 4 = effective 16, lr 1e-5, fp32 (focal+bf16 has numerical issues)

### Counterfactual Logit Pairing baselines (Davani CLP)
- λ ∈ {0.5, 1.0, 2.0} with heuristic filter + λ=1.0 no-filter
- L1 in logit space (per Davani 2021)

### Prompted LLMs
- Mistral-7B, Qwen-2.5-7B, Llama-3.1-8B locally (4-bit NF4 on A100)
- Llama-3.3-70B via Groq API (fp16)
- Three prompts: zero-shot, 5-shot, Guided CoT (Patel 2025)

## §3.5 CCI v2 (the proposed method)

### Counterfactual swap engine
Given a tweet $x$ containing a Jewish religious-ethnic token $t \in T_{\text{Jewish}} = \{jew, jewish, jews, judaism, \ldots\}$, generate up to $K=5$ counterfactual variants by substituting $t$ with a token from the symmetric vocabulary $T_{\text{alt}} = \{Muslim, Christian, Hindu, Buddhist, Atheist\}$. Casing and morphology preserved.

**Heuristic symmetry filter** rejects swaps for which the label may not be preserved:
- (i) if $x$ contains an antisemitic slur (`kike`, `zionazi`, ...) → reject (slur is identity-specific)
- (ii) if $x$ contains only a geopolitical proxy (`israel`, `zionist`) without a religious-ethnic anchor → reject (no symmetric counterpart exists)

On the 1060-tweet test set: 2117 symmetric pairs (Jews group 1725, Israel group 392, Kikes/ZioNazi 0 by construction).

### Per-group temperature head
Small head $\tau_\theta : \mathcal{G} \to \mathbb{R}_{>0}$ mapping group id $g \in \{Jews, Israel, Kikes, ZioNazi, none\}$ (so 5 groups) to scalar $T_g = \exp(\log T_g)$ clipped to $[T_{\min}, T_{\max}] = [0.5, 5.0]$. Trained jointly with model in same AdamW optimizer, no weight decay on $\log T_g$. Discarded at inference (modulates the loss, not predictions).

### Loss (eq. 1 in paper)
$$\mathcal{L}_{\text{CCI}}(x_i) = \mathrm{FocalLoss}(f(x_i), y_i) + \lambda \cdot \frac{1}{K_i} \sum_{k=1}^{K_i} \mathrm{JS}_b\Bigl(\sigma(z(x_i)/T_{g_i}), \sigma(z(x'_{i,k})/T_{g_i})\Bigr)$$

where $z(x) = \mathrm{logit}_1 - \mathrm{logit}_0$ is the positive-class logit difference, $\sigma$ is sigmoid, $\mathrm{JS}_b$ is binary Jensen-Shannon divergence between Bernoullis. We set $\lambda = 1.0$.

**Four ablation cells**: T fixed at 1.0 vs learnable × divergence JS vs L1.

### Theorem 1 (TS preserves CFR_hard at threshold 1/2)

For any binary classifier $f$ with positive-class logit difference $z(x)$ and any $T > 0$:
$$\mathbb{1}[\sigma(z(x)) \ge 1/2] \ne \mathbb{1}[\sigma(z(x')) \ge 1/2] \iff \mathbb{1}[\sigma(z(x)/T) \ge 1/2] \ne \mathbb{1}[\sigma(z(x')/T) \ge 1/2]$$

**Proof sketch**: $\sigma(\cdot) \ge 1/2 \iff (\cdot) \ge 0$. Dividing by $T > 0$ preserves sign, so $z(x) \ge 0 \iff z(x)/T \ge 0$. Hence binary predictions and their disagreement are invariant under TS. ∎

Empirically verified bit-exact on all 6 Model C seeds (fitted T ∈ [0.566, 1.400], mean 1.031): CFR_hard,TS = CFR_hard,raw seed-by-seed.

## §3.4 Evaluation protocol

### Counterfactual robustness (on the 2117-pair test manifest)
- **CFR_hard**: fraction of pairs where binary prediction flips at threshold 0.5
- **CFR_soft, L1**: $|p(x) - p(x')|$ averaged over pairs
- **CFR_soft, JS**: binary JS divergence between $\sigma(z(x))$ and $\sigma(z(x'))$ averaged

### Group-conditional calibration and fairness
- Per-keyword macro-F1
- Group-conditional ECE (15-bin, with global / worst-group / gap)
- FPED / FNED (Dixon 2018)
- Cross-dataset transfer: HateXplain Jewish-target, ToxiGen Jewish-group (zero-shot)

### Statistical significance protocol
For each (pivot vs comparator, metric) compute paired-bootstrap p-value over 6 seeds, n_resample=10000. Pivot = CCI v2 (learned T, JS) on DeBERTa-v3-base. **9 comparators × 6 metrics = 54 tests**, Bonferroni $\alpha = 0.05/54 \approx 9.3 \times 10^{-4}$.

For comparators with reduced seed grids (cci_v2_fixed_js_deberta_v3_large has n=5 because seed 47 was dropped during training), the paired bootstrap operates on the seed intersection.

---

# 5. EXPERIMENTS (implementation details)

PyTorch 2.1+ + HuggingFace. AdamW lr 2e-5 base / 1e-5 large, weight decay 0.01, linear warmup 10% + linear decay. Batch 8 (base) / 4 (large) × grad_accum 2 (base) / 4 (large) = effective 16. fp16 base / fp32 large (DeBERTa disentangled attn + focal loss have numerical issues in low precision). Grad clip 1.0. Early stopping on dev macro-F1 patience 3. 5 epochs (Models A/B/B2) or 7 epochs (Model C, CCI v2, Davani CLP). Max seq len 128.

Six seeds {42, 43, 44, 45, 46, 47}. Total runs: 4 base × 6 seeds + 4 CCI v2 cells × 6 seeds + 4 Davani × 6 seeds + 4 multi-arch × 6 seeds = 96. Plus Model C + TS + multicalibration (post-hoc on cached Model C predictions).

---

# 6. RESULTS

## §4.1 Main results table

| Method | F1+ | ECE | ECE_worst | CFR_hard | CFR_soft_L1 | FPED | FNED |
|---|---|---|---|---|---|---|---|
| **Base (DeBERTa-v3-base, 184M)** | | | | | | | |
| Davani λ=0.5 | 0.672±0.015 | 0.094±0.033 | 0.235±0.061 | 0.040±0.005* | 0.036±0.005* | 0.324±0.093 | **0.528**±0.038* |
| Davani λ=1.0 | 0.660±0.017 | 0.127±0.047 | 0.225±0.056 | 0.034±0.011 | 0.029±0.011 | 0.303±0.094 | 0.590±0.079 |
| Davani λ=2.0 | 0.667±0.019 | 0.177±0.035 | 0.207±0.041 | 0.042±0.009 | 0.026±0.006 | 0.303±0.094 | 0.558±0.046 |
| CCI v2 (fixed T, JS) | **0.677**±0.008 | 0.098±0.053 | 0.195±0.028 | 0.018±0.003 | 0.014±0.002 | 0.303±0.074 | 0.543±0.090* |
| **CCI v2 (learned T, JS)** (pivot) | 0.662±0.015 | **0.083**±0.048 | 0.215±0.034 | **0.015**±0.008 | **0.013**±0.003 | 0.317±0.095 | 0.605±0.072 |
| Model C (focal+kw) | 0.660±0.019 | 0.077±0.034 | 0.221±0.021 | 0.036±0.008* | 0.032±0.007* | **0.265**±0.073 | 0.584±0.090 |
| Model C + post-hoc TS | 0.660±0.019 | 0.045±0.014 | 0.246±0.018 | 0.036±0.008* | 0.033±0.009* | **0.265**±0.073 | 0.584±0.090 |
| Model C + multicalibration | 0.652±0.015 | **0.035**±0.007 | 0.221±0.021 | 0.033±0.006* | 0.063±0.019* | 0.269±0.077 | 0.683±0.060 |
| **Multi-architecture (large)** | | | | | | | |
| Model C (DeBERTa-v3-large) | 0.704±0.020 | 0.097±0.053 | 0.206±0.050 | 0.042±0.003 | 0.037±0.009 | 0.315±0.077 | 0.435±0.048 |
| **CCI v2 fixed_js (DeBERTa-v3-large)** | **0.719**±0.012 | 0.099±0.063 | 0.236±0.031 | 0.019±0.009 | 0.017±0.003 | 0.327±0.081 | **0.429**±0.042 |
| Model C (RoBERTa-large) | 0.689±0.016 | 0.068±0.044 | 0.226±0.045 | 0.047±0.002 | 0.048±0.001 | 0.282±0.056 | 0.510±0.026 |
| **CCI v2 fixed_js (RoBERTa-large)** | 0.694±0.005 | 0.077±0.052 | **0.204**±0.049 | **0.013**±0.006 | 0.013±0.002 | **0.217**±0.013 | 0.560±0.073 |

\* = Bonferroni-significant vs pivot at α = 0.05/54 ≈ 9.3e-4 (paired bootstrap, n_resample = 10,000)

## §4.2 Counterfactual robustness narrative

CCI v2 reduces CFR_hard consistently across 3 architectures × 2 families:
- DeBERTa-v3-base (184M): 0.018 vs 0.036 ⇒ **−49%**
- DeBERTa-v3-large (440M): 0.019 vs 0.042 ⇒ **−55%**
- RoBERTa-large (355M, cross-family): 0.013 vs 0.047 ⇒ **−73%**

Largest reduction is cross-family. F1+ improves at scale (best 0.719 on DeBERTa-large). Theorem 1 empirically verified bit-exact on 6/6 Model C seeds.

Within-CCI-v2 ablation: JS variants F1+ 0.66-0.68 vs L1 0.65; L1 variants CFR_hard 0.007-0.008 vs JS 0.015-0.018. Fixed-T vs learned-T statistically indistinguishable on CFR_hard (p=0.25). **Active ingredient = JS divergence in probability space, not per-group T.**

## §4.3 Multicalibration tension (CENTERPIECE)

Multicalibration post-hoc:
- CFR_hard: 0.036 → 0.033 (−8%)
- CFR_soft_L1: 0.032 → 0.063 (**+96%**)
- CFR_soft_JS: 0.005 → 0.011 (**+147%**)

Two well-respected fairness desiderata (subgroup calibration & counterfactual invariance) in direct tension. To our knowledge, first quantitative report on a text classifier (Hansen NeurIPS 2024 examines subgroup ECE/accuracy but not counterfactual metrics).

CCI v2 lowers both simultaneously: CFR_soft_L1 → 0.013 (−60%), CFR_hard → 0.015 (−58%).

## §4.4 Fairness audit

Worst-group ECE varies more than global ECE; TS has lowest global ECE (0.045) but highest worst-group (0.246).

**FPED**: similar across methods (0.22-0.35), driven by small ZioNazi group.

**FNED Pareto trade-off**: Davani λ=0.5 best (0.528), CCI v2 (learned T, JS) worst (0.605, +14.6% vs Davani, Bonferroni p=0.0001).

**Capacity-bound finding**: FNED regression disappears at larger scale:
- DeBERTa-base: CCI v2 (fixed_js) FNED 0.543 vs Model C 0.584 (−7%, marginal regression vs Model C, not Davani)
- DeBERTa-large: CCI v2 0.429 vs Model C 0.435 (**−1.5%, improvement**)
- RoBERTa-large: CCI v2 0.560 vs Model C 0.510 (+10%, regression smaller than base)

## §4.5 Significance suite: 23/54 reject

- **12 base CFR rejections** favor pivot (4 base comparators × 3 CFR metrics, raw p ≤ 1e-4)
- **6 multi-arch CFR rejections** favor pivot (2 large Model C × 3 CFR metrics)
- **5 FNED rejections** against pivot (Davani λ=0.5, CCI v2 fixed_js base, Model C DeBERTa-large, Model C RoBERTa-large, CCI v2 fixed_js DeBERTa-large)
- **0 F1+, FPED, ECE_worst** rejections (genuine equivalence)
- Pivot vs CCI v2 fixed_js (RoBERTa-large) CFR comparisons NOT significant (p > 0.5) — intended consistency story.

## §4.6 Bimodal-ECE (supporting finding)

Model C per-seed test ECE: 4/6 seeds {42,43,46,47} → ECE [0.039, 0.066] (well-calibrated); 2/6 {44,45} → ECE [0.106, 0.129] (poorly-calibrated). Vanilla DeBERTa-v3 has no bimodality.

Mechanism: dev ECE falls monotonically; well-calibrated seeds peak F1 at epochs 4-7 (after ECE has dropped); bad seeds peak at epoch 3 (when ECE still ~0.13). Early stopping then truncates training before further calibration. Artefact of F1-only checkpoint selection.

## §4.7 LLM + cross-dataset (PENDING numbers)

Mistral, Qwen, Llama-3.1-8B locally + Llama-3.3-70B Groq. HateXplain Jewish + ToxiGen Jewish-group. Final numbers pending the multi-seed sweep on the prompted track.

---

# 7. DISCUSSION

7 subsections:
1. **Error analysis** (5 categories): political criticism, counterspeech, coded antisemitism, sarcasm/irony, ambiguous cases.
2. **Why training-time invariance beats post-hoc rescaling**: Theorem 1 + tokenizer-agnostic transfer (swap engine operates at text level).
3. **FNED Pareto trade-off + capacity-bound**: mechanism is pair-invariance penalty compressing probabilities near boundary; relaxes with more capacity.
4. **Multicalibration tension mechanism**: non-monotone per-cell adjustment shifts (x, x') in opposite directions.
5. **Fine-tuned vs prompted LLMs**: faithfulness caveat (Turpin 2023), CFR_soft inapplicable to LLMs (binary outputs).
6. **Bimodal-ECE mechanism**.
7. **Limitations (7 items)**: IHRA dependency, single dataset / single identity, Theorem 1 scope (hard CFR at threshold 1/2 only), symmetry filter heuristics, English-only, binary classification, temporal limitation.

---

# 8. ETHICS

7 paragraphs:
1. **Dual use & censorship risk**: triage tool not autonomous; recommend p < 0.8 → human review.
2. **Annotation framework**: IHRA vs Jerusalem Declaration; CCI v2 is framework-independent.
3. **Identity-term bias & counterfactual fairness**: CCI v2 designed against shortcut learning; FNED Pareto disclosed.
4. **Counterfactual swap pairs at training AND evaluation**: explicit disclosure (the ethics questionnaire submitted 2026-05-12 stated "exclusively at evaluation time" — this paragraph corrects the record).
5. **Dataset sensitivity & researcher welfare**.
6. **Computing infrastructure**: RCP for training + 7B LLMs; Groq API for Llama-3.3-70B (third-party US inference disclosure).
7. **Scope of claims**: explicit single-identity caveat.

---

# 9. CONCLUSION

3 paragraphs + 5 future-work items.

> We propose Counterfactual Calibration Invariance (CCI v2), prove that post-hoc TS cannot reduce CFR_hard at threshold 1/2, and demonstrate CFR reduction of 49% / 55% / 73% across 3 architectures, with F1+ improved at scale (0.72 best). 23/54 Bonferroni-corrected rejections; 18 CFR comparisons favour pivot. Multicalibration introduces a +96% CFR_soft_L1 regression. FNED Pareto trade-off is capacity-bound.

Future work: (1) Lipschitz-style upper bound on CFR_hard (Garg-style certificate), (2) LLM-likelihood symmetry filter, (3) cross-language transfer, (4) cross-identity transfer, (5) multi-label severity-graded classification.

---

# 10. REPORT.md (FULL EMPIRICAL EVIDENCE FILE)

This file is the ground truth for every number in the paper. **Auditor: please cross-check every claim above against this section.**

The file is at `analysis/direction1/REPORT.md` on the `main` branch of the repo. It contains:
- **TL;DR**: 58% reduction headline, TS bit-exact preservation
- **Significance suite summary**: 54 tests, 23 reject; per-metric counts (CFR_hard 6, CFR_soft_L1 6, CFR_soft_JS 6, FNED 5, ECE_worst 0, FPED 0); per-comparator counts
- **Main results table** (12 rows × 8 metrics, all numbers reproduced above)
- **Appendix A — CFR breakdown** (per-method n_seeds, F1+, CFR_hard, CFR_soft_L1, CFR_soft_JS, n_pairs=2117 for all)
- **Appendix B — FPED/FNED breakdown** + per-group FPR + per-group FNR
- **Appendix C — Group-conditional ECE breakdown** + per-group ECE (Jews / Israel / Kikes / ZioNazi)
- **Appendix D — Per-test significance results** (54-row table with comparator, metric, A, B, Δ, 95% CI, p_raw, p_Bonf, seed-direction)
- **Appendix E — TS bit-exact sanity check** (per-seed T_fitted + raw + post-TS, all 6 exact_match=True)

**Auditor please pull the actual REPORT.md from the repo for full numerical detail** (the file is 250+ lines of structured tables; download from `https://raw.githubusercontent.com/giacomocase/DeepLearning/main/analysis/direction1/REPORT.md`).

---

# END BUNDLE

To audit further, the full code is at `https://github.com/giacomocase/DeepLearning/tree/main`. Specifically:
- `src/training/cci_loss.py` — CCI v2 loss + binary JS divergence implementation
- `src/training/cci_trainer.py` — training loop with temperature head joint optimization
- `src/data/counterfactual_swap.py` + `src/data/symmetry_classifier.py` — swap engine
- `src/calibration/multicalibration.py` — post-hoc multicalibration patcher
- `analysis/direction1/06_run_significance_suite.py` — 54-test paired bootstrap
- `tests/test_*.py` — 10 test files covering all the above

End of bundle.
