#!/usr/bin/env bash
# Multi-architecture robustness sweep — DeBERTa-v3-large + RoBERTa-large.
#
# Tests whether the CCI v2 effect generalises beyond DeBERTa-v3-base (184M)
# to two larger and architecturally different backbones:
#
#   * DeBERTa-v3-LARGE  (440M)  — same family, larger scale
#   * RoBERTa-LARGE     (355M)  — different family (standard attention)
#
# For each architecture we run:
#   * Model C (focal + identity-mask)     -> reproduces the bimodal-ECE baseline
#   * CCI v2 (fixed T, JS)                -> the pivot for counterfactual robustness
#
# Total: 2 cells × 2 architectures × 6 seeds = 24 jobs on H100-80g.
# With 10 concurrent H100, wall-clock ETA ~2-3 hours.
#
# Pre-requisites:
#   * Same as submit_cci_v2.sh — repo on /home/<user>, image on registry,
#     home + scratch PVCs accessible.
#
# Usage:
#   bash rcp/submit_multiarch.sh                     # full 24-job sweep
#   bash rcp/submit_multiarch.sh smoke               # single 1-epoch smoke
#   SEEDS="42 43" bash rcp/submit_multiarch.sh       # restricted seed list
#   bash rcp/submit_multiarch.sh full default       # alternate node pool

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-default}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

# Smoke = 4 cells × seed 42 × 1 epoch (uses *_smoke.yaml configs with epochs=1,
# patience=1). Exercises all 4 architecture × method combinations to catch
# numerical issues (e.g. bf16+focal underflow, dtype mismatches) BEFORE the
# full 24-job sweep.
# Wall clock ETA: ~5-10 min with 4 H100 concurrent.
if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting multi-arch smoke: 4 cells × seed 42 × 1 epoch..."
  declare -a SMOKE_CELLS=(
    "model_c_deberta_v3_large_smoke:train.py"
    "model_c_roberta_large_smoke:train.py"
    "cci_v2_fixed_js_deberta_v3_large_smoke:train_cci.py"
    "cci_v2_fixed_js_roberta_large_smoke:train_cci.py"
  )
  for smoke_cell in "${SMOKE_CELLS[@]}"; do
    IFS=':' read -r SMOKE_CFG SMOKE_SCRIPT <<< "${smoke_cell}"
    SMOKE_JOB_NAME="ma-${SMOKE_CFG//_/-}-s42"
    SMOKE_JOB_NAME="$(echo "${SMOKE_JOB_NAME}" | cut -c 1-50)"
    bash "${HERE}/submit.sh" "${SMOKE_JOB_NAME}" \
      "python3 scripts/${SMOKE_SCRIPT} --config configs/${SMOKE_CFG}.yaml --no-wandb --seed 42" \
      "${NODE_POOL}"
  done
  echo ""
  echo "Smoke submitted: ${#SMOKE_CELLS[@]} cells. Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep ma-.*-smoke'"
  exit 0
fi

# (config, train_script) pairs — Model C uses train.py, CCI v2 uses train_cci.py
declare -a CELLS=(
  "model_c_deberta_v3_large:train.py"
  "model_c_roberta_large:train.py"
  "cci_v2_fixed_js_deberta_v3_large:train_cci.py"
  "cci_v2_fixed_js_roberta_large:train_cci.py"
)

for SEED in "${SEEDS[@]}"; do
  for cell in "${CELLS[@]}"; do
    IFS=':' read -r CFG SCRIPT <<< "${cell}"
    JOB_NAME="ma-${CFG//_/-}-s${SEED}"
    # Run:ai max name length is ~50 chars — truncate just in case
    JOB_NAME="$(echo "${JOB_NAME}" | cut -c 1-50)"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/${SCRIPT} --config configs/${CFG}.yaml --no-wandb --seed ${SEED}" \
      "${NODE_POOL}"
  done
done

echo ""
echo "Multi-arch sweep submitted: ${#CELLS[@]} cells × ${#SEEDS[@]} seeds = $((${#CELLS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep ma-'"
