#!/usr/bin/env bash
# Calibration bake-off (Layer 1 of the audacity plan).
#
# Submits the 6 calibration configurations × 6 seeds = 36 training runs to
# RCP a100-40g. With concurrent A100s available, wall-clock should be under
# 30 minutes; even sequentially it is ~3 hours.
#
# Strategies:
#   1. calib_F1                  -> vanilla best-F1 early stop (baseline)
#   2. calib_composite_a05       -> argmax(F1 - 0.5 * ECE)
#   3. calib_composite_a10       -> argmax(F1 - 1.0 * ECE)
#   4. calib_composite_a20       -> argmax(F1 - 2.0 * ECE)
#   5. calib_TS                  -> Temperature scaling (Guo et al. ICML 2017)
#   6. calib_SWA                 -> Stochastic Weight Averaging last 3 epochs
#
# Usage:
#   bash rcp/submit_calibration.sh                       # all 6 configs × 3 seeds
#   SEEDS="42 43 44 45 46 47" bash rcp/submit_calibration.sh   # 6 seeds (paper)
#
# All configurations share DeBERTa-v3-base + focal loss + keyword masking
# (Model C). The variable being ablated is checkpoint-selection only.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_POOL="${1:-a100-40g}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

CONFIGS=(
  "calibration_F1"
  "calibration_composite_a05"
  "calibration_composite_a10"
  "calibration_composite_a20"
  "calibration_TS"
  "calibration_SWA"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    bash "${HERE}/submit.sh" "seed${SEED}-${CFG//_/-}" \
      "python3 scripts/train.py --config configs/${CFG}.yaml --no-wandb --seed ${SEED}" \
      "${NODE_POOL}"
  done
done

echo ""
echo "Calibration bake-off submitted: ${#CONFIGS[@]} strategies × ${#SEEDS[@]} seeds = $((${#CONFIGS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor with: watch -n 30 runai list"
