# Counterfactual flip-rate as the loss-aligned metric

CCI v2 explicitly minimises pair-invariance of `σ(z/T_g)` under identity-term
swap. ECE measures something different (marginal calibration). This directory
re-evaluates all 48 trained methods + Model C raw + Model C + post-hoc TS +
Model C + post-hoc multicalibration on the metric the loss was designed for:
**counterfactual flip-rate** (CFR), in three flavours (hard / L1 / JS).

## Pipeline

```
01_build_swap_manifest.py   →  cfr_cache/_shared/test_swaps_manifest.parquet
02_forward_pass.py          →  cfr_cache/<exp>_seed<N>/{test_originals,
                                                          test_swap_logits,
                                                          dev_originals*}.parquet
03_compute_cfr.py           →  results/{per_checkpoint_cfr.parquet,
                                         cfr_summary.csv,
                                         main_table_cfr.md,
                                         ts_sanity_check.csv}
```

`*` dev_originals.parquet is written only for `model_c_deberta_focal_mask`,
since it is the basis for fitting post-hoc TS and multicalibration.

## Run order (first time)

On the **jumphost** (CPU only):

```bash
cd ~/antisemitism-detection
python3 analysis/01_build_swap_manifest.py
```

Then submit the 54-job forward-pass sweep on **RCP**:

```bash
bash rcp/submit_forward_pass.sh
# Or: bash rcp/submit_forward_pass.sh smoke   # one job for sanity
```

When all jobs are `Succeeded`, run the CPU aggregation on the jumphost:

```bash
python3 analysis/03_compute_cfr.py
cat analysis/results/main_table_cfr.md
cat analysis/results/ts_sanity_check.csv
```

## What to look at first in the results

1. **`ts_sanity_check.csv`**: every row should have `exact_match = True`. This
   verifies the structural claim that CFR_hard is invariant under post-hoc TS
   for binary classification at threshold 0.5. If any row says False, there's
   a bug; investigate before trusting the rest.
2. **`main_table_cfr.md`**: paper-ready table. Rows of interest:
   - `Model C (focal+kw)`: raw baseline.
   - `Model C + post-hoc TS`: CFR_hard MUST be identical to the row above.
   - `CCI v2 (learned T, JS)`: pivot method we hope reduces CFR_hard.
   - `Davani λ=0.5`: strongest training-time competitor we already
     identified.

## Outputs cached on /scratch (not in repo)

```
$SCRATCH_ROOT/cfr_cache/
├── _shared/
│   ├── test_swaps_manifest.parquet
│   └── test_swaps_manifest_meta.json
└── <experiment>_seed<N>/
    ├── manifest.json
    ├── test_originals.parquet
    ├── test_swap_logits.parquet
    └── dev_originals.parquet      # Model C only
```

The `cfr_cache/` directory is **not gitignored** (it's outside the repo
entirely, on the scratch PVC). Total size: ~2 GB across 54 checkpoints.

## Method label vocabulary

| Internal name                       | Display name                  |
|-------------------------------------|-------------------------------|
| `davani_lambda05`                   | Davani λ=0.5                  |
| `davani_lambda10`                   | Davani λ=1.0                  |
| `davani_lambda20`                   | Davani λ=2.0                  |
| `davani_lambda10_nofilt`            | Davani λ=1.0 (no filter)      |
| `cci_v2_fixed_l1`                   | CCI v2 (fixed T, L1)          |
| `cci_v2_fixed_js`                   | CCI v2 (fixed T, JS)          |
| `cci_v2_learned_l1`                 | CCI v2 (learned T, L1)        |
| `cci_v2_learned_js`                 | CCI v2 (learned T, JS)        |
| `model_c_deberta_focal_mask`        | Model C (focal+kw)            |
| `model_c_plus_ts`                   | Model C + post-hoc TS         |
| `model_c_plus_multicalibration`     | Model C + multicalibration    |

## Pipeline scripts (all shipped)

- `04_compute_fped_fned.py`: Dixon AIES 2018 metrics on the 4 GoldStandard
  keyword groups
- `05_compute_group_ece.py`: per-group ECE (15 bins)
- `06_run_significance_suite.py`: paired-by-seed bootstrap + Holm-Bonferroni
  step-down across two pre-specified families (54 main + 6 within-architecture
  transfer tests) + exact two-sided sign-test backup
- `07_build_main_table.py`: assemble F1+ × ECE × CFR × FPED/FNED headline
- `09_write_report.py`: REPORT.md (08 reserved for the cross-dataset transfer
  block that depends on the external cross-dataset CSVs and is not yet implemented)
