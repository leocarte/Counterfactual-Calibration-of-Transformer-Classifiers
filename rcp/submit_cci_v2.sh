#!/usr/bin/env bash
# CCI v2 ablation sweep — Phase 3 of the training-experiments branch.
#
# Submits 4 ablation cells × 6 seeds = 24 training runs to RCP a100-40g.
#
# Cells (all at λ=1.0, central setting):
#   1. cci_v2_fixed_l1      -> T fixed, L1 divergence (≈ CLP-with-focal collapse)
#   2. cci_v2_fixed_js      -> T fixed, JS divergence (divergence-only effect)
#   3. cci_v2_learned_l1    -> T_g learned, L1 divergence (per-group T effect)
#   4. cci_v2_learned_js    -> T_g learned, JS divergence (the "real" CCI v2)
#
# Seeds: {42, 43, 44, 45, 46, 47}.
#
# Usage:
#   bash rcp/submit_cci_v2.sh                     # full 24-job sweep
#   bash rcp/submit_cci_v2.sh smoke               # single 1-epoch smoke
#   SEEDS="42 43" bash rcp/submit_cci_v2.sh       # restricted seed list

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-a100-40g}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting single CCI v2 smoke job (configs/cci_v2_smoke.yaml, seed=42)..."
  bash "${HERE}/submit.sh" "te-cci-v2-smoke" \
    "python3 scripts/train_cci.py --config configs/cci_v2_smoke.yaml --no-wandb --seed 42" \
    "${NODE_POOL}"
  exit 0
fi

CONFIGS=(
  "cci_v2_fixed_l1"
  "cci_v2_fixed_js"
  "cci_v2_learned_l1"
  "cci_v2_learned_js"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    JOB_NAME="te-${CFG//_/-}-s${SEED}"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/train_cci.py --config configs/${CFG}.yaml --no-wandb --seed ${SEED}" \
      "${NODE_POOL}"
  done
done

echo ""
echo "CCI v2 sweep submitted: ${#CONFIGS[@]} cells × ${#SEEDS[@]} seeds = $((${#CONFIGS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami)'"
