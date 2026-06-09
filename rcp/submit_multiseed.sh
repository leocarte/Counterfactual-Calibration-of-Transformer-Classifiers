#!/usr/bin/env bash
# Multi-seed transformer sweep for statistical significance.
#
# Runs each of the 4 models with 3 different random seeds (42, 43, 44) so we
# can report mean ± std in the paper. The single-seed full ablations are
# helpful but a 0.008 macro_f1 gap between models could easily be noise at
# this test-set size (187 positives).
#
# Total: 3 seeds × 4 models = 12 jobs, parallel on a100-40g, ETA ~15-20 min.
#
# train.py now accepts --seed N (overrides cfg.seed AND appends _seed{N} to
# the experiment_name so outputs don't collide).
#
# Usage: bash rcp/submit_multiseed.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_POOL="${1:-a100-40g}"

# Seeds can be overridden via env var, e.g. `SEEDS="45 46 47" bash rcp/submit_multiseed.sh`
# Default: the initial 3-seed sweep.
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 43 44})

for SEED in "${SEEDS[@]}"; do
  bash "${HERE}/submit.sh" "seed${SEED}-model-a-roberta" \
    "python3 scripts/train.py --config configs/roberta.yaml --no-wandb --seed ${SEED}" \
    "${NODE_POOL}"

  bash "${HERE}/submit.sh" "seed${SEED}-model-b-deberta" \
    "python3 scripts/train.py --config configs/deberta.yaml --no-wandb --seed ${SEED}" \
    "${NODE_POOL}"

  bash "${HERE}/submit.sh" "seed${SEED}-model-b2-hatebert" \
    "python3 scripts/train.py --config configs/hatebert.yaml --no-wandb --seed ${SEED}" \
    "${NODE_POOL}"

  bash "${HERE}/submit.sh" "seed${SEED}-model-c-deberta-focal-mask" \
    "python3 scripts/train.py --config configs/deberta_focal_mask.yaml --no-wandb --seed ${SEED}" \
    "${NODE_POOL}"
done

echo ""
echo "Multi-seed sweep submitted: ${#SEEDS[@]} seeds × 4 models = $((${#SEEDS[@]} * 4)) jobs."
echo "Monitor with: watch -n 30 runai list"
