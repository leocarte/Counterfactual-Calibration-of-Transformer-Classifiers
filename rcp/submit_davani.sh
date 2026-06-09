#!/usr/bin/env bash
# Davani CLP sweep — Phase 1 of the training-experiments branch.
#
# Submits 4 configurations × 6 seeds = 24 training runs to RCP a100-40g.
#
# Configurations:
#   1. davani_lambda05            -> CLP, λ=0.5, heuristic symmetry filter
#   2. davani_lambda10            -> CLP, λ=1.0, heuristic symmetry filter (central)
#   3. davani_lambda20            -> CLP, λ=2.0, heuristic symmetry filter
#   4. davani_lambda10_nofilt     -> CLP, λ=1.0, NO filter (asymmetric-pair ablation)
#
# Seeds: {42, 43, 44, 45, 46, 47} (matches existing multi-seed sweeps).
#
# Usage:
#   bash rcp/submit_davani.sh                     # full sweep (24 jobs)
#   bash rcp/submit_davani.sh smoke               # single 1-epoch smoke job
#   SEEDS="42 43" bash rcp/submit_davani.sh       # restricted seed list
#   bash rcp/submit_davani.sh full a100-80g       # full sweep on 80g node pool

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-a100-40g}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting single smoke job (configs/davani_smoke.yaml, 1 epoch, seed=42)..."
  bash "${HERE}/submit.sh" "te-davani-smoke" \
    "python3 scripts/train_clp.py --config configs/davani_smoke.yaml --no-wandb --seed 42" \
    "${NODE_POOL}"
  echo ""
  echo "Tail: runai logs te-davani-smoke -p course-ee-559-\$(whoami)"
  exit 0
fi

CONFIGS=(
  "davani_lambda05"
  "davani_lambda10"
  "davani_lambda20"
  "davani_lambda10_nofilt"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    JOB_NAME="te-${CFG//_/-}-s${SEED}"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/train_clp.py --config configs/${CFG}.yaml --no-wandb --seed ${SEED}" \
      "${NODE_POOL}"
  done
done

echo ""
echo "Davani sweep submitted: ${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds = $((${#CONFIGS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami)'"
