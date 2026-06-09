# Counterfactual flip-rate as the loss-aligned metric

_Generated: 2026-05-14T17:15:44.005950+00:00_

_Pivot method for significance testing: **CCI v2 (fixed T, JS)**_

## TL;DR

**CCI v2 (fixed T, JS)** achieves CFR_hard = 0.0183 on the test set (mean over 6 seeds), a **48.8% reduction** vs Model C (focal+kw, CFR_hard = 0.0358). On F1+, the pivot reaches 0.6767 vs Model C's 0.6597. Post-hoc Temperature Scaling on Model C cannot reduce CFR_hard (verified bit-exact, 6/6 seeds: CFR_hard_TS = 0.0358 = CFR_hard_raw). Post-hoc multicalibration applied per-input-group reduces CFR_hard slightly to 0.0330 but at the cost of higher CFR_soft (see Table 1). The strongest training-time competitor (Davani λ=0.5) reaches CFR_hard = 0.0403 — still 2.2× the pivot's value.

## Sanity check: TS preserves CFR_hard

All **6/6** Model C seeds have CFR_hard exactly preserved under post-hoc Temperature Scaling (T fitted on dev). Fitted T range: 0.566 – 1.400 (mean 1.031; the wide range reflects the bimodal-ECE phenomenon in Model C training). The structural claim *"post-hoc monotone rescaling preserves CFR_hard at threshold 0.5"* is thus both proved analytically and verified empirically.

## Significance suite (Bonferroni-corrected)

**Family `main`** (pivot = CCI v2 (fixed T, JS)): 54 paired-bootstrap tests, α_Bonferroni = 0.05/54 = 0.0009259. Bonferroni rejects: **23**; Holm-Bonferroni rejects: **23** (each row also reports an exact sign-test p-value).

Bonferroni rejects by metric: cfr_hard: 6, cfr_soft_js: 6, cfr_soft_l1: 6, ece_worst_group: 2, fned: 3, fped: 0.

Bonferroni rejects by comparator: CCI v2 fixed_js (DeBERTa-v3-large): 1, CCI v2 fixed_js (RoBERTa-large): 0, CCI v2 (learned T, JS): 1, Davani λ=0.5: 3, Model C (focal+kw): 3, Model C (DeBERTa-v3-large): 4, Model C + multicalibration: 4, Model C + post-hoc TS: 4, Model C (RoBERTa-large): 3.

**Family `within_arch_transfer`** (pivot = CCI v2 fixed_js (DeBERTa-v3-large)): 6 paired-bootstrap tests, α_Bonferroni = 0.05/6 = 0.008333. Bonferroni rejects: **5**; Holm-Bonferroni rejects: **6** (each row also reports an exact sign-test p-value).

Bonferroni rejects by metric: cfr_hard: 2, cfr_soft_js: 1, cfr_soft_l1: 2.

Bonferroni rejects by comparator: Model C (DeBERTa-v3-large): 2, Model C (RoBERTa-large): 3.

Full per-test details in `significance_summary.md`.

## Main results table

# CFR-evaluation — Main results table

Mean ± std across 6 seeds. **\*** denotes significance vs the pivot (*CCI v2 (fixed T, JS)*) at Bonferroni-corrected α = 0.05/60 ≈ 0.0008333. Lower is better for ECE, CFR, FPED, FNED. Higher is better for F1+.

| Method | F1+ | ECE | ECE worst | CFR hard | CFR soft (L1) | CFR soft (JS) | FPED | FNED |
|---|---|---|---|---|---|---|---|---|
| Davani λ=0.5 | 0.6722 ± 0.0148 | 0.0943 ± 0.0327 | 0.2350 ± 0.0614 | 0.0403 ± 0.0049* | 0.0355 ± 0.0052* | 0.0050 ± 0.0022* | 0.3242 ± 0.0930 | 0.5283 ± 0.0383 |
| Davani λ=1.0 | 0.6602 ± 0.0172 | 0.1273 ± 0.0469 | 0.2253 ± 0.0562 | 0.0337 ± 0.0114 | 0.0289 ± 0.0105 | 0.0033 ± 0.0016 | 0.3030 ± 0.0939 | 0.5897 ± 0.0793 |
| Davani λ=2.0 | 0.6674 ± 0.0188 | 0.1773 ± 0.0349 | 0.2067 ± 0.0413 | 0.0419 ± 0.0085 | 0.0261 ± 0.0062 | 0.0022 ± 0.0013 | 0.3030 ± 0.0939 | 0.5584 ± 0.0462 |
| Davani λ=1.0 (no filter) | 0.6606 ± 0.0252 | 0.1618 ± 0.0550 | 0.2307 ± 0.0536 | 0.0411 ± 0.0098 | 0.0282 ± 0.0039 | 0.0023 ± 0.0011 | 0.3485 ± 0.0684 | 0.5191 ± 0.0667 |
| CCI v2 (fixed T, L1) | 0.6511 ± 0.0117 | 0.1995 ± 0.0194 | 0.2180 ± 0.0213 | 0.0067 ± 0.0021 | 0.0033 ± 0.0005 | 0.0001 ± 0.0001 | 0.3485 ± 0.0684 | 0.6806 ± 0.0639 |
| **CCI v2 (fixed T, JS)** (pivot) | 0.6767 ± 0.0079 | 0.0981 ± 0.0526 | 0.1953 ± 0.0283 | 0.0183 ± 0.0028 | 0.0141 ± 0.0016 | 0.0008 ± 0.0004 | 0.3030 ± 0.0742 | 0.5434 ± 0.0896 |
| CCI v2 (learned T, L1) | 0.6468 ± 0.0103 | 0.1863 ± 0.0120 | 0.2080 ± 0.0162 | 0.0078 ± 0.0023 | 0.0036 ± 0.0003 | 0.0001 ± 0.0000 | 0.3030 ± 0.0469 | 0.6759 ± 0.0227 |
| CCI v2 (learned T, JS) | 0.6615 ± 0.0150 | 0.0830 ± 0.0480 | 0.2152 ± 0.0338 | 0.0150 ± 0.0075 | 0.0130 ± 0.0026 | 0.0011 ± 0.0006 | 0.3166 ± 0.0945 | 0.6053 ± 0.0718* |
| Model C (focal+kw) | 0.6597 ± 0.0194 | 0.0772 ± 0.0342 | 0.2213 ± 0.0208 | 0.0358 ± 0.0083* | 0.0322 ± 0.0065* | 0.0045 ± 0.0018* | 0.2652 ± 0.0728 | 0.5839 ± 0.0900 |
| Model C + post-hoc TS | 0.6597 ± 0.0194 | 0.0454 ± 0.0135 | 0.2461 ± 0.0179* | 0.0358 ± 0.0083* | 0.0329 ± 0.0089* | 0.0046 ± 0.0014* | 0.2652 ± 0.0728 | 0.5839 ± 0.0900 |
| Model C + multicalibration | 0.6519 ± 0.0151 | 0.0354 ± 0.0070 | 0.2213 ± 0.0208 | 0.0330 ± 0.0059* | 0.0630 ± 0.0187* | 0.0111 ± 0.0051* | 0.2692 ± 0.0770 | 0.6834 ± 0.0600* |
| Model C (DeBERTa-v3-large) | 0.7041 ± 0.0202 | 0.0966 ± 0.0531 | 0.2058 ± 0.0495 | 0.0417 ± 0.0034* | 0.0373 ± 0.0086* | 0.0063 ± 0.0046* | 0.3150 ± 0.0767 | 0.4352 ± 0.0478* |
| Model C (RoBERTa-large) | 0.6894 ± 0.0156 | 0.0678 ± 0.0438 | 0.2255 ± 0.0446 | 0.0465 ± 0.0023* | 0.0479 ± 0.0007* | 0.0106 ± 0.0041* | 0.2824 ± 0.0563 | 0.5098 ± 0.0264 |
| CCI v2 fixed_js (DeBERTa-v3-large) | 0.7194 ± 0.0121 | 0.0986 ± 0.0630 | 0.2361 ± 0.0310* | 0.0187 ± 0.0093 | 0.0165 ± 0.0026 | 0.0023 ± 0.0017 | 0.3273 ± 0.0813 | 0.4285 ± 0.0420 |
| CCI v2 fixed_js (RoBERTa-large) | 0.6935 ± 0.0053 | 0.0766 ± 0.0519 | 0.2044 ± 0.0493 | 0.0126 ± 0.0064 | 0.0130 ± 0.0019 | 0.0011 ± 0.0005 | 0.2174 ± 0.0133 | 0.5602 ± 0.0728 |

---

## Appendix A — CFR breakdown

# CFR-evaluation main table — Counterfactual Flip Rate (CFR)
Mean ± std across seeds (n=6 unless otherwise noted).

| Method | n_seeds | F1+ | CFR_hard | CFR_soft_L1 | CFR_soft_JS | n_pairs |
|---|---|---|---|---|---|---|
| Davani λ=0.5 | 6 | 0.6722 ± 0.0148 | 0.0403 ± 0.0049 | 0.0355 ± 0.0052 | 0.0050 ± 0.0022 | 2117 |
| Davani λ=1.0 | 6 | 0.6602 ± 0.0172 | 0.0337 ± 0.0114 | 0.0289 ± 0.0105 | 0.0033 ± 0.0016 | 2117 |
| Davani λ=2.0 | 6 | 0.6674 ± 0.0188 | 0.0419 ± 0.0085 | 0.0261 ± 0.0062 | 0.0022 ± 0.0013 | 2117 |
| Davani λ=1.0 (no filter) | 6 | 0.6606 ± 0.0252 | 0.0411 ± 0.0098 | 0.0282 ± 0.0039 | 0.0023 ± 0.0011 | 2117 |
| CCI v2 (fixed T, L1) | 6 | 0.6511 ± 0.0117 | 0.0067 ± 0.0021 | 0.0033 ± 0.0005 | 0.0001 ± 0.0001 | 2117 |
| CCI v2 (fixed T, JS) | 6 | 0.6767 ± 0.0079 | 0.0183 ± 0.0028 | 0.0141 ± 0.0016 | 0.0008 ± 0.0004 | 2117 |
| CCI v2 (learned T, L1) | 6 | 0.6468 ± 0.0103 | 0.0078 ± 0.0023 | 0.0036 ± 0.0003 | 0.0001 ± 0.0000 | 2117 |
| CCI v2 (learned T, JS) | 6 | 0.6615 ± 0.0150 | 0.0150 ± 0.0075 | 0.0130 ± 0.0026 | 0.0011 ± 0.0006 | 2117 |
| Model C (focal+kw) | 6 | 0.6597 ± 0.0194 | 0.0358 ± 0.0083 | 0.0322 ± 0.0065 | 0.0045 ± 0.0018 | 2117 |
| Model C + post-hoc TS | 6 | 0.6597 ± 0.0194 | 0.0358 ± 0.0083 | 0.0329 ± 0.0089 | 0.0046 ± 0.0014 | 2117 |
| Model C + multicalibration | 6 | 0.6519 ± 0.0151 | 0.0330 ± 0.0059 | 0.0630 ± 0.0187 | 0.0111 ± 0.0051 | 2117 |

---

## Appendix B — FPED / FNED breakdown

# CFR-evaluation main table — FPED / FNED (Dixon AIES 2018)
Mean ± std across seeds (n=6 unless otherwise noted). Lower is better — both FPED and FNED measure the spread of group-conditional rates across the 4 GoldStandard2024 identity-keyword groups (Jews, Israel, Kikes, ZioNazi).

Per-group test counts: **Jews** 599 (72 pos), **Israel** 396 (65 pos), **Kikes** 20 (9 pos), **ZioNazi** 45 (41 pos).

| Method | n_seeds | FPED | FNED |
|---|---|---|---|
| Davani λ=0.5 | 6 | 0.3242 ± 0.0930 | 0.5283 ± 0.0383 |
| Davani λ=1.0 | 6 | 0.3030 ± 0.0939 | 0.5897 ± 0.0793 |
| Davani λ=2.0 | 6 | 0.3030 ± 0.0939 | 0.5584 ± 0.0462 |
| Davani λ=1.0 (no filter) | 6 | 0.3485 ± 0.0684 | 0.5191 ± 0.0667 |
| CCI v2 (fixed T, L1) | 6 | 0.3485 ± 0.0684 | 0.6806 ± 0.0639 |
| CCI v2 (fixed T, JS) | 6 | 0.3030 ± 0.0742 | 0.5434 ± 0.0896 |
| CCI v2 (learned T, L1) | 6 | 0.3030 ± 0.0469 | 0.6759 ± 0.0227 |
| CCI v2 (learned T, JS) | 6 | 0.3166 ± 0.0945 | 0.6053 ± 0.0718 |
| Model C (focal+kw) | 6 | 0.2652 ± 0.0728 | 0.5839 ± 0.0900 |
| Model C + post-hoc TS | 6 | 0.2652 ± 0.0728 | 0.5839 ± 0.0900 |
| Model C + multicalibration | 6 | 0.2692 ± 0.0770 | 0.6834 ± 0.0600 |

## Per-group FPR (false positive rate)

| Method | Jews | Israel | Kikes | Zionazi |
|---|---|---|---|---|
| Davani λ=0.5 | 0.0446 ± 0.0099 | 0.0821 ± 0.0172 | 0.3333 ± 0.0939 | 0.0417 ± 0.1021 |
| Davani λ=1.0 | 0.0373 ± 0.0116 | 0.0720 ± 0.0136 | 0.3030 ± 0.0939 | 0.0000 ± 0.0000 |
| Davani λ=2.0 | 0.0471 ± 0.0112 | 0.0846 ± 0.0156 | 0.3030 ± 0.0939 | 0.0000 ± 0.0000 |
| Davani λ=1.0 (no filter) | 0.0610 ± 0.0191 | 0.1032 ± 0.0338 | 0.3485 ± 0.0684 | 0.0000 ± 0.0000 |
| CCI v2 (fixed T, L1) | 0.0310 ± 0.0075 | 0.0811 ± 0.0062 | 0.3485 ± 0.0684 | 0.0000 ± 0.0000 |
| CCI v2 (fixed T, JS) | 0.0395 ± 0.0197 | 0.0720 ± 0.0304 | 0.3030 ± 0.0742 | 0.0000 ± 0.0000 |
| CCI v2 (learned T, L1) | 0.0392 ± 0.0077 | 0.0791 ± 0.0102 | 0.3030 ± 0.0469 | 0.0000 ± 0.0000 |
| CCI v2 (learned T, JS) | 0.0326 ± 0.0163 | 0.0705 ± 0.0436 | 0.3182 ± 0.0953 | 0.0417 ± 0.1021 |
| Model C (focal+kw) | 0.0459 ± 0.0154 | 0.0695 ± 0.0287 | 0.2727 ± 0.0813 | 0.0417 ± 0.1021 |
| Model C + post-hoc TS | 0.0459 ± 0.0154 | 0.0695 ± 0.0287 | 0.2727 ± 0.0813 | 0.0417 ± 0.1021 |
| Model C + multicalibration | 0.0304 ± 0.0094 | 0.0498 ± 0.0192 | 0.2727 ± 0.0813 | 0.0417 ± 0.1021 |

## Per-group FNR (false negative rate)

| Method | Jews | Israel | Kikes | Zionazi |
|---|---|---|---|---|
| Davani λ=0.5 | 0.5324 ± 0.0399 | 0.3872 ± 0.0226 | 0.1296 ± 0.0454 | 0.0041 ± 0.0100 |
| Davani λ=1.0 | 0.6019 ± 0.0728 | 0.3974 ± 0.0562 | 0.2222 ± 0.0703 | 0.0122 ± 0.0134 |
| Davani λ=2.0 | 0.5625 ± 0.0428 | 0.3487 ± 0.0650 | 0.2037 ± 0.0836 | 0.0041 ± 0.0100 |
| Davani λ=1.0 (no filter) | 0.5231 ± 0.0673 | 0.3282 ± 0.0967 | 0.1111 ± 0.0703 | 0.0041 ± 0.0100 |
| CCI v2 (fixed T, L1) | 0.6806 ± 0.0639 | 0.3615 ± 0.0233 | 0.1667 ± 0.0609 | 0.0000 ± 0.0000 |
| CCI v2 (fixed T, JS) | 0.5556 ± 0.0938 | 0.3846 ± 0.0667 | 0.1481 ± 0.0574 | 0.0122 ± 0.0134 |
| CCI v2 (learned T, L1) | 0.6759 ± 0.0227 | 0.3590 ± 0.0186 | 0.1852 ± 0.0574 | 0.0000 ± 0.0000 |
| CCI v2 (learned T, JS) | 0.6134 ± 0.0802 | 0.4103 ± 0.0605 | 0.1852 ± 0.0907 | 0.0081 ± 0.0126 |
| Model C (focal+kw) | 0.5880 ± 0.0937 | 0.3974 ± 0.0711 | 0.1852 ± 0.0574 | 0.0041 ± 0.0100 |
| Model C + post-hoc TS | 0.5880 ± 0.0937 | 0.3974 ± 0.0711 | 0.1852 ± 0.0574 | 0.0041 ± 0.0100 |
| Model C + multicalibration | 0.6875 ± 0.0574 | 0.4308 ± 0.0496 | 0.1852 ± 0.0574 | 0.0041 ± 0.0100 |

---

## Appendix C — Group-conditional ECE breakdown

# CFR-evaluation main table — Group-conditional ECE
Mean ± std across seeds (n=6 unless otherwise noted). ECE computed with 15 equal-width bins (Guo ICML 2017). Per-group ECE uses the dataset's keyword grouping; worst-group ECE is the max across the 4 groups.

| Method | n_seeds | ECE_global | ECE_worst | ECE_gap |
|---|---|---|---|---|
| Davani λ=0.5 | 6 | 0.0943 ± 0.0327 | 0.2350 ± 0.0614 | 0.1406 ± 0.0597 |
| Davani λ=1.0 | 6 | 0.1273 ± 0.0469 | 0.2253 ± 0.0562 | 0.0980 ± 0.0658 |
| Davani λ=2.0 | 6 | 0.1773 ± 0.0349 | 0.2067 ± 0.0413 | 0.0294 ± 0.0186 |
| Davani λ=1.0 (no filter) | 6 | 0.1618 ± 0.0550 | 0.2307 ± 0.0536 | 0.0689 ± 0.0415 |
| CCI v2 (fixed T, L1) | 6 | 0.1995 ± 0.0194 | 0.2180 ± 0.0213 | 0.0185 ± 0.0123 |
| CCI v2 (fixed T, JS) | 6 | 0.0981 ± 0.0526 | 0.1953 ± 0.0283 | 0.0972 ± 0.0502 |
| CCI v2 (learned T, L1) | 6 | 0.1863 ± 0.0120 | 0.2080 ± 0.0162 | 0.0217 ± 0.0148 |
| CCI v2 (learned T, JS) | 6 | 0.0830 ± 0.0480 | 0.2152 ± 0.0338 | 0.1322 ± 0.0503 |
| Model C (focal+kw) | 6 | 0.0772 ± 0.0342 | 0.2213 ± 0.0208 | 0.1441 ± 0.0541 |
| Model C + post-hoc TS | 6 | 0.0454 ± 0.0135 | 0.2461 ± 0.0179 | 0.2008 ± 0.0225 |
| Model C + multicalibration | 6 | 0.0354 ± 0.0070 | 0.2213 ± 0.0208 | 0.1858 ± 0.0155 |

## Per-group ECE breakdown

| Method | Jews | Israel | Kikes | Zionazi |
|---|---|---|---|---|
| Davani λ=0.5 | 0.0919 ± 0.0329 | 0.1130 ± 0.0316 | 0.2350 ± 0.0614 | 0.0612 ± 0.0174 |
| Davani λ=1.0 | 0.1311 ± 0.0513 | 0.1461 ± 0.0440 | 0.2173 ± 0.0606 | 0.0789 ± 0.0247 |
| Davani λ=2.0 | 0.1861 ± 0.0322 | 0.1869 ± 0.0361 | 0.2052 ± 0.0424 | 0.0880 ± 0.0069 |
| Davani λ=1.0 (no filter) | 0.1651 ± 0.0566 | 0.1732 ± 0.0572 | 0.2230 ± 0.0534 | 0.0919 ± 0.0267 |
| CCI v2 (fixed T, L1) | 0.2085 ± 0.0189 | 0.2095 ± 0.0177 | 0.1878 ± 0.0465 | 0.1050 ± 0.0193 |
| CCI v2 (fixed T, JS) | 0.0985 ± 0.0571 | 0.1173 ± 0.0491 | 0.1755 ± 0.0571 | 0.0883 ± 0.0160 |
| CCI v2 (learned T, L1) | 0.2006 ± 0.0142 | 0.1928 ± 0.0126 | 0.1898 ± 0.0299 | 0.0956 ± 0.0144 |
| CCI v2 (learned T, JS) | 0.0784 ± 0.0491 | 0.1026 ± 0.0523 | 0.2152 ± 0.0338 | 0.0725 ± 0.0250 |
| Model C (focal+kw) | 0.0765 ± 0.0322 | 0.0943 ± 0.0411 | 0.2213 ± 0.0208 | 0.0719 ± 0.0307 |
| Model C + post-hoc TS | 0.0374 ± 0.0073 | 0.0713 ± 0.0206 | 0.2461 ± 0.0179 | 0.0712 ± 0.0372 |
| Model C + multicalibration | 0.0420 ± 0.0048 | 0.0596 ± 0.0116 | 0.2213 ± 0.0208 | 0.0315 ± 0.0215 |

---

## Appendix D — Per-test significance results

# Significance suite — pivot: CCI v2 (fixed T, JS)

Paired bootstrap (Dror et al. ACL 2018), n_resample=10,000. Bonferroni-corrected α = 0.05/54 = 0.0009 for the main family. Tests organised in two pre-specified hypothesis families: **main** (pivot vs comparators on all 6 metrics), **within_arch_transfer** (large-arch CCI vs large-arch Model C on the 3 CFR metrics only). Each family is Bonferroni-corrected separately AND Holm-corrected per family. Reported per row: bootstrap p-value (`p_boot`), exact sign-test p-value (`p_sign`, minimum achievable under H0 on direction agreement), `n_common` (paired seeds used; 5 vs 6 when one comparator dropped a seed). Reject marker: **B** = Bonferroni, **H** = Holm-Bonferroni. Δ = pivot − comparator; negative Δ on CFR/FPED/FNED/ECE means pivot is better.


## Family: `main` (pivot: CCI v2 (fixed T, JS))

54 tests; Bonferroni rejects: 23; Holm-Bonferroni rejects: 23.

| Comparator | Metric | A (pivot) | B (comp) | Δ (A−B) | 95% CI | p_boot | p_sign | Bonf | Holm | n A<B / n |
|---|---|---|---|---|---|---|---|---|---|---|
| CCI v2 (learned T, JS) | cfr_hard | 0.0183 | 0.0150 | +0.0033 | [-0.0024, +0.0083] | 0.248 | 0.2188 |   |   | 1/6 |
| CCI v2 (learned T, JS) | cfr_soft_l1 | 0.0141 | 0.0130 | +0.0011 | [-0.0016, +0.0039] | 0.4226 | 1 |   |   | 3/6 |
| CCI v2 (learned T, JS) | cfr_soft_js | 0.0008 | 0.0011 | -0.0002 | [-0.0006, +0.0000] | 0.1014 | 0.6875 |   |   | 4/6 |
| CCI v2 (learned T, JS) | fped | 0.3030 | 0.3166 | -0.0136 | [-0.0909, +0.0606] | 0.8666 | 0.6875 |   |   | 2/6 |
| CCI v2 (learned T, JS) | fned | 0.5434 | 0.6053 | -0.0619 | [-0.1198, -0.0180] | 0.0008 | 0.2188 | **B** | **H** | 5/6 |
| CCI v2 (learned T, JS) | ece_worst_group | 0.1953 | 0.2152 | -0.0198 | [-0.0486, +0.0096] | 0.1968 | 0.6875 |   |   | 4/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Davani λ=0.5 | cfr_hard | 0.0183 | 0.0403 | -0.0220 | [-0.0250, -0.0191] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Davani λ=0.5 | cfr_soft_l1 | 0.0141 | 0.0355 | -0.0214 | [-0.0257, -0.0164] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Davani λ=0.5 | cfr_soft_js | 0.0008 | 0.0050 | -0.0042 | [-0.0056, -0.0027] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Davani λ=0.5 | fped | 0.3030 | 0.3242 | -0.0211 | [-0.1029, +0.0758] | 0.651 | 0.2188 |   |   | 3/6 |
| Davani λ=0.5 | fned | 0.5434 | 0.5283 | +0.0150 | [-0.0405, +0.0643] | 0.5394 | 1 |   |   | 3/6 |
| Davani λ=0.5 | ece_worst_group | 0.1953 | 0.2350 | -0.0396 | [-0.0836, +0.0181] | 0.1572 | 0.2188 |   |   | 5/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C (focal+kw) | cfr_hard | 0.0183 | 0.0358 | -0.0175 | [-0.0229, -0.0121] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (focal+kw) | cfr_soft_l1 | 0.0141 | 0.0322 | -0.0180 | [-0.0231, -0.0123] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (focal+kw) | cfr_soft_js | 0.0008 | 0.0045 | -0.0037 | [-0.0048, -0.0027] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (focal+kw) | fped | 0.3030 | 0.2652 | +0.0379 | [+0.0076, +0.0682] | 0.0322 | 0.03125 |   |   | 0/6 |
| Model C (focal+kw) | fned | 0.5434 | 0.5839 | -0.0405 | [-0.1412, +0.0358] | 0.43 | 0.6875 |   |   | 3/6 |
| Model C (focal+kw) | ece_worst_group | 0.1953 | 0.2213 | -0.0259 | [-0.0536, +0.0042] | 0.0968 | 0.2188 |   |   | 5/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C + post-hoc TS | cfr_hard | 0.0183 | 0.0358 | -0.0175 | [-0.0229, -0.0121] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + post-hoc TS | cfr_soft_l1 | 0.0141 | 0.0329 | -0.0188 | [-0.0258, -0.0117] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + post-hoc TS | cfr_soft_js | 0.0008 | 0.0046 | -0.0038 | [-0.0049, -0.0028] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + post-hoc TS | fped | 0.3030 | 0.2652 | +0.0379 | [+0.0076, +0.0682] | 0.0322 | 0.03125 |   |   | 0/6 |
| Model C + post-hoc TS | fned | 0.5434 | 0.5839 | -0.0405 | [-0.1412, +0.0358] | 0.43 | 0.6875 |   |   | 3/6 |
| Model C + post-hoc TS | ece_worst_group | 0.1953 | 0.2461 | -0.0508 | [-0.0699, -0.0334] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C + multicalibration | cfr_hard | 0.0183 | 0.0330 | -0.0146 | [-0.0202, -0.0096] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + multicalibration | cfr_soft_l1 | 0.0141 | 0.0630 | -0.0489 | [-0.0645, -0.0363] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + multicalibration | cfr_soft_js | 0.0008 | 0.0111 | -0.0102 | [-0.0144, -0.0071] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C + multicalibration | fped | 0.3030 | 0.2692 | +0.0338 | [+0.0035, +0.0641] | 0.0322 | 0.03125 |   |   | 0/6 |
| Model C + multicalibration | fned | 0.5434 | 0.6834 | -0.1401 | [-0.2350, -0.0474] | 0.0002 | 0.2188 | **B** | **H** | 5/6 |
| Model C + multicalibration | ece_worst_group | 0.1953 | 0.2213 | -0.0259 | [-0.0536, +0.0042] | 0.0968 | 0.2188 |   |   | 5/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C (DeBERTa-v3-large) | cfr_hard | 0.0183 | 0.0417 | -0.0234 | [-0.0266, -0.0198] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (DeBERTa-v3-large) | cfr_soft_l1 | 0.0141 | 0.0373 | -0.0232 | [-0.0281, -0.0169] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (DeBERTa-v3-large) | cfr_soft_js | 0.0008 | 0.0063 | -0.0055 | [-0.0089, -0.0025] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (DeBERTa-v3-large) | fped | 0.3030 | 0.3150 | -0.0120 | [-0.0959, +0.0676] | 0.8076 | 0.6875 |   |   | 2/6 |
| Model C (DeBERTa-v3-large) | fned | 0.5434 | 0.4352 | +0.1082 | [+0.0450, +0.1771] | 0.0001 | 0.03125 | **B** | **H** | 0/6 |
| Model C (DeBERTa-v3-large) | ece_worst_group | 0.1953 | 0.2058 | -0.0104 | [-0.0559, +0.0325] | 0.6636 | 1 |   |   | 3/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C (RoBERTa-large) | cfr_hard | 0.0183 | 0.0465 | -0.0282 | [-0.0309, -0.0255] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (RoBERTa-large) | cfr_soft_l1 | 0.0141 | 0.0479 | -0.0337 | [-0.0353, -0.0321] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (RoBERTa-large) | cfr_soft_js | 0.0008 | 0.0106 | -0.0097 | [-0.0126, -0.0063] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (RoBERTa-large) | fped | 0.3030 | 0.2824 | +0.0207 | [-0.0548, +0.0943] | 0.5708 | 0.6875 |   |   | 2/6 |
| Model C (RoBERTa-large) | fned | 0.5434 | 0.5098 | +0.0335 | [-0.0347, +0.1041] | 0.3694 | 0.6875 |   |   | 2/6 |
| Model C (RoBERTa-large) | ece_worst_group | 0.1953 | 0.2255 | -0.0302 | [-0.0764, +0.0199] | 0.2286 | 0.2188 |   |   | 5/6 |
|  |  |  |  |  |  |  |  |  |  |  |
| CCI v2 fixed_js (DeBERTa-v3-large) | cfr_hard | 0.0182 | 0.0187 | -0.0005 | [-0.0091, +0.0070] | 0.9362 | 1 |   |   | 3/5 |
| CCI v2 fixed_js (DeBERTa-v3-large) | cfr_soft_l1 | 0.0142 | 0.0165 | -0.0023 | [-0.0046, +0.0000] | 0.0586 | 0.375 |   |   | 4/5 |
| CCI v2 fixed_js (DeBERTa-v3-large) | cfr_soft_js | 0.0009 | 0.0023 | -0.0014 | [-0.0028, -0.0001] | 0.0368 | 1 |   |   | 3/5 |
| CCI v2 fixed_js (DeBERTa-v3-large) | fped | 0.2909 | 0.3273 | -0.0364 | [-0.1273, +0.0727] | 0.5754 | 0.375 |   |   | 3/5 |
| CCI v2 fixed_js (DeBERTa-v3-large) | fned | 0.5152 | 0.4285 | +0.0868 | [+0.0125, +0.1611] | 0.0242 | 0.375 |   |   | 1/5 |
| CCI v2 fixed_js (DeBERTa-v3-large) | ece_worst_group | 0.2004 | 0.2361 | -0.0357 | [-0.0714, -0.0067] | 0.0001 | 0.0625 | **B** | **H** | 5/5 |
|  |  |  |  |  |  |  |  |  |  |  |
| CCI v2 fixed_js (RoBERTa-large) | cfr_hard | 0.0183 | 0.0126 | +0.0057 | [+0.0017, +0.0094] | 0.0074 | 0.2188 |   |   | 1/6 |
| CCI v2 fixed_js (RoBERTa-large) | cfr_soft_l1 | 0.0141 | 0.0130 | +0.0011 | [-0.0003, +0.0029] | 0.1574 | 0.6875 |   |   | 2/6 |
| CCI v2 fixed_js (RoBERTa-large) | cfr_soft_js | 0.0008 | 0.0011 | -0.0003 | [-0.0007, +0.0003] | 0.3392 | 0.2188 |   |   | 5/6 |
| CCI v2 fixed_js (RoBERTa-large) | fped | 0.3030 | 0.2174 | +0.0856 | [+0.0266, +0.1377] | 0.0018 | 0.2188 |   |   | 1/6 |
| CCI v2 fixed_js (RoBERTa-large) | fned | 0.5434 | 0.5602 | -0.0168 | [-0.1042, +0.0717] | 0.6984 | 1 |   |   | 3/6 |
| CCI v2 fixed_js (RoBERTa-large) | ece_worst_group | 0.1953 | 0.2044 | -0.0090 | [-0.0530, +0.0337] | 0.6954 | 1 |   |   | 3/6 |
|  |  |  |  |  |  |  |  |  |  |  |

## Family: `within_arch_transfer` (pivot: CCI v2 fixed_js (DeBERTa-v3-large))

6 tests; Bonferroni rejects: 5; Holm-Bonferroni rejects: 6.

| Comparator | Metric | A (pivot) | B (comp) | Δ (A−B) | 95% CI | p_boot | p_sign | Bonf | Holm | n A<B / n |
|---|---|---|---|---|---|---|---|---|---|---|
| Model C (DeBERTa-v3-large) | cfr_hard | 0.0187 | 0.0420 | -0.0233 | [-0.0304, -0.0141] | 0.0001 | 0.0625 | **B** | **H** | 5/5 |
| Model C (DeBERTa-v3-large) | cfr_soft_l1 | 0.0165 | 0.0401 | -0.0237 | [-0.0276, -0.0180] | 0.0001 | 0.0625 | **B** | **H** | 5/5 |
| Model C (DeBERTa-v3-large) | cfr_soft_js | 0.0023 | 0.0072 | -0.0049 | [-0.0099, -0.0005] | 0.0372 | 0.375 |   | **H** | 4/5 |
|  |  |  |  |  |  |  |  |  |  |  |
| Model C (RoBERTa-large) | cfr_hard | 0.0126 | 0.0465 | -0.0339 | [-0.0391, -0.0279] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (RoBERTa-large) | cfr_soft_l1 | 0.0130 | 0.0479 | -0.0349 | [-0.0363, -0.0335] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
| Model C (RoBERTa-large) | cfr_soft_js | 0.0011 | 0.0106 | -0.0095 | [-0.0126, -0.0059] | 0.0001 | 0.03125 | **B** | **H** | 6/6 |
|  |  |  |  |  |  |  |  |  |  |  |

---

## Appendix E — TS-preserves-CFR_hard sanity check (raw)

```
 seed  T_fitted  cfr_hard_raw  cfr_hard_ts  exact_match
   42  1.399579      0.032121     0.032121         True
   43  1.244211      0.029287     0.029287         True
   44  0.601919      0.044402     0.044402         True
   45  0.566012      0.046292     0.046292         True
   46  1.388256      0.025508     0.025508         True
   47  0.984683      0.037317     0.037317         True
```
