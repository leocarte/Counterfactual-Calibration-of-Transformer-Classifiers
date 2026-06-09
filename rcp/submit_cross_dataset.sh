#!/usr/bin/env bash
# Cross-dataset transfer evaluation on the CCI v2 pivot checkpoints (and
# optionally on the same-architecture Model C checkpoints for a 4-row table).
#
# WHY: the original cross-dataset run (commit 7a7e0fc) used Model C only,
# but the paper's `methodology.tex` cross-dataset block claims "our best
# fine-tuned model", which is CCI v2 fixed_js. This script re-runs the
# same evaluate.py pipeline on the CCI v2 checkpoints that already exist
# on RCP scratch from the multi-architecture sweep.
#
# Usage:
#   bash rcp/submit_cross_dataset.sh                    # CCI v2 base only, 6 seeds
#   bash rcp/submit_cross_dataset.sh large              # base + CCI v2 DeBERTa-large (n=5) + CCI v2 RoBERTa-large
#   bash rcp/submit_cross_dataset.sh large_only         # ONLY the two large variants (skip base; useful if base already done)
#   bash rcp/submit_cross_dataset.sh all                # base + larges + same-arch Model C (29 jobs total)
#
# Prereqs:
#   - data/external/hatexplain_jewish.csv and toxigen_jewish.csv exist (committed)
#   - CCI v2 checkpoints exist at:
#       /scratch/$USER/antisemitism-results/checkpoints/cci_v2_fixed_js_seed{42..47}/best_model.pt
#       /scratch/$USER/antisemitism-results/checkpoints/cci_v2_fixed_js_deberta_v3_large_seed{42..46}/best_model.pt
#       /scratch/$USER/antisemitism-results/checkpoints/cci_v2_fixed_js_roberta_large_seed{42..47}/best_model.pt
#
# evaluate.py already handles wrapped CCI v2 checkpoints (the {model,
# temperature_head} unwrap is in scripts/evaluate.py:58-67 since commit b1eeabe).

set -euo pipefail

MODE="${1:-base}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"
SCRATCH_RESULTS="/scratch/${RCP_USERNAME}/antisemitism-results"

submit_one() {
  local config_name="$1"
  local checkpoint_name="$2"
  local seed="$3"
  local config_path="configs/${config_name}.yaml"
  local ckpt_path="${SCRATCH_RESULTS}/checkpoints/${checkpoint_name}_seed${seed}/best_model.pt"
  local out_dir="${SCRATCH_RESULTS}/metrics/cross_dataset_${checkpoint_name}_seed${seed}"
  local job_name="xds-${checkpoint_name}-s${seed}"
  # Run:ai requires lowercase + hyphens, max 50 chars
  job_name="$(echo "${job_name}" | tr '_A-Z' '-a-z' | cut -c 1-50)"
  bash "${HERE}/submit.sh" "${job_name}" \
    "python3 scripts/evaluate.py --config ${config_path} --checkpoint ${ckpt_path} --output-dir ${out_dir}" \
    a100-40g
}

# CCI v2 base (the headline pivot — 6 seeds). Skip if large_only.
if [[ "${MODE}" != "large_only" ]]; then
  echo "=== Submitting cross-dataset eval on CCI v2 base (6 seeds) ==="
  for seed in 42 43 44 45 46 47; do
    submit_one "cci_v2_fixed_js" "cci_v2_fixed_js" "${seed}"
  done
fi

if [[ "${MODE}" == "large" || "${MODE}" == "large_only" || "${MODE}" == "all" ]]; then
  # CCI v2 DeBERTa-large (n=5 — seed 47 dropped per REPORT.md convention)
  echo ""
  echo "=== Submitting cross-dataset eval on CCI v2 DeBERTa-large (5 seeds) ==="
  for seed in 42 43 44 45 46; do
    submit_one "cci_v2_fixed_js_deberta_v3_large" \
               "cci_v2_fixed_js_deberta_v3_large" "${seed}"
  done

  # CCI v2 RoBERTa-large (cross-family — 6 seeds)
  echo ""
  echo "=== Submitting cross-dataset eval on CCI v2 RoBERTa-large (6 seeds) ==="
  for seed in 42 43 44 45 46 47; do
    submit_one "cci_v2_fixed_js_roberta_large" \
               "cci_v2_fixed_js_roberta_large" "${seed}"
  done
fi

if [[ "${MODE}" == "all" ]]; then

  # Same-architecture Model C on the two larges, to complete the 4-row table:
  #   Base       : Model C vs CCI v2     (Model C already done in commit 7a7e0fc)
  #   DeBERTa-L  : Model C vs CCI v2
  #   RoBERTa-L  : Model C vs CCI v2
  echo ""
  echo "=== Submitting cross-dataset eval on Model C large variants (11 seeds) ==="
  for seed in 42 43 44 45 46 47; do
    submit_one "model_c_deberta_v3_large" "model_c_deberta_v3_large" "${seed}"
  done
  for seed in 42 43 44 45 46 47; do
    submit_one "model_c_roberta_large" "model_c_roberta_large" "${seed}"
  done
fi

echo ""
echo "Cross-dataset eval jobs submitted (mode=${MODE})."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep xds-'"
echo ""
echo "When all Succeeded, pull results back to laptop:"
echo "  scp -r ${RCP_USERNAME}@jumphost.rcp.epfl.ch:/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/${RCP_USERNAME}/antisemitism-results/metrics/cross_dataset_* results/metrics_from_rcp/"
