#!/usr/bin/env bash
# Second group-conditional calibrator vs counterfactual invariance, on the 6
# already-trained Model C seeds. Shows the soft-CFR inflation the paper reports
# for HKRR multicalibration GENERALIZES to a second calibrator (per-group
# temperature scaling) — the novelty 8->9 lever. NO training: reads the Model C
# checkpoints from scratch and only forward-passes + fits + measures.
#
# PREREQUISITES (on scratch):
#   - the shared swap manifest: build once with
#       python3 analysis/direction1/01_build_swap_manifest.py
#   - the Model C checkpoints at
#       /scratch/$USER/antisemitism-results/checkpoints/model_c_deberta_focal_mask_seed{42..47}/best_model.pt
#
# Usage:
#   bash rcp/submit_calibrator_cfr.sh                              # per_group_temp, 6 seeds (the point)
#   CALIBRATOR=multicalibration bash rcp/submit_calibrator_cfr.sh  # replicate the paper's HKRR result
#   SEEDS="42 43" bash rcp/submit_calibrator_cfr.sh                # restricted seeds
#
# Each job is a few minutes (no training). Aggregate the per-seed metrics.json
# for the paper: CFR_soft pre->post and the inflation %.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"
NODE_POOL="${NODE_POOL:-default}"
CALIBRATOR="${CALIBRATOR:-per_group_temp}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})
SCRATCH="/scratch/${RCP_USERNAME}/antisemitism-results"

for SEED in "${SEEDS[@]}"; do
  CKPT="${SCRATCH}/checkpoints/model_c_deberta_focal_mask_seed${SEED}/best_model.pt"
  OUT="${SCRATCH}/calibrator_cfr/${CALIBRATOR}_seed${SEED}"
  bash "${HERE}/submit.sh" "te-cal-${CALIBRATOR//_/-}-s${SEED}" \
    "python3 scripts/eval_calibrator_cfr.py --checkpoint ${CKPT} --config configs/deberta_focal_mask.yaml --calibrator ${CALIBRATOR} --output-dir ${OUT}" \
    "${NODE_POOL}"
done

echo ""
echo "Calibrator-CFR sweep submitted (${CALIBRATOR}, ${#SEEDS[@]} seeds, pool=${NODE_POOL})."
echo "Each job: forward-pass + fit + CFR_soft pre/post over the swap manifest (~few min, no training)."
echo "Watch: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep te-cal-'"
