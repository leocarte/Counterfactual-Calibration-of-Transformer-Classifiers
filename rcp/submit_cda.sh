#!/usr/bin/env bash
# CDA baseline sweep — counterfactual data augmentation, no consistency loss.
#
# Trains the Model C recipe (DeBERTa-v3-base + focal + keyword masking) on the
# original train split PLUS the admissible identity-swapped copies as extra
# labeled rows (label copied from the source). Answers the reviewer question
# "is CCI v2 just more counterfactual data?" — WOAH2026_READINESS.md item 2.
#
# Submits 1 config × 6 seeds = 6 training runs to RCP a100-40g.
#
# Seeds: {42, 43, 44, 45, 46, 47} (same convention as rcp/submit_cci_v2.sh).
#
# Usage:
#   bash rcp/submit_cda.sh                     # full 6-seed sweep
#   bash rcp/submit_cda.sh smoke               # single 1-epoch smoke
#   SEEDS="42 43" bash rcp/submit_cda.sh       # restricted seed list

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-default}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting single CDA smoke job (configs/cda_deberta_base_smoke.yaml, seed=42)..."
  bash "${HERE}/submit.sh" "te-cda-smoke" \
    "python3 scripts/train.py --config configs/cda_deberta_base_smoke.yaml --no-wandb --seed 42" \
    "${NODE_POOL}"
  exit 0
fi

for SEED in "${SEEDS[@]}"; do
  JOB_NAME="te-cda-deberta-base-s${SEED}"
  bash "${HERE}/submit.sh" "${JOB_NAME}" \
    "python3 scripts/train.py --config configs/cda_deberta_base.yaml --no-wandb --seed ${SEED}" \
    "${NODE_POOL}"
done

echo ""
echo "CDA sweep submitted: ${#SEEDS[@]} seeds × 1 config = ${#SEEDS[@]} jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami)'"
