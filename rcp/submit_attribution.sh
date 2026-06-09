#!/usr/bin/env bash
# Integrated Gradients attribution fanout for §4.5 paper figure.
#
# Submits 9 jobs (3 models × 3 seeds) computing per-example IG attribution
# on 500 test examples. Output: 9 JSON files in results/interpretability/.
#
# Models compared:
#   * Model B (vanilla DeBERTa-v3-base)
#   * Model C (focal + identity-mask)
#   * CCI v2 (fixed T, JS divergence) — our method
#
# Wall clock: ~15-30 min per job on a100-40g; 9 parallel ~30 min total.
#
# Usage:
#   bash rcp/submit_attribution.sh                       # full 9-job sweep
#   bash rcp/submit_attribution.sh full a100-80g         # alternate node pool
#   SEEDS="42" bash rcp/submit_attribution.sh            # single-seed test

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_POOL="${1:-default}"
SEEDS=(${SEEDS:-42 43 44})
USER_NAME="${RCP_USERNAME:-$(whoami)}"

# (tag, checkpoint-dir-prefix) — checkpoint dir is constructed as
# /scratch/<user>/antisemitism-results/checkpoints/<prefix><seed>/best_model.pt
declare -a MODELS=(
  "modelb:model_b_deberta_v3_base_seed"             # configs/deberta.yaml experiment_name
  "modelc:model_c_deberta_focal_mask_seed"          # configs/deberta_focal_mask.yaml
  "cciv2:cci_v2_fixed_js_seed"                      # configs/cci_v2_fixed_js.yaml
)

for spec in "${MODELS[@]}"; do
  IFS=':' read -r TAG CKPT_PREFIX <<< "${spec}"
  for SEED in "${SEEDS[@]}"; do
    CKPT="/scratch/${USER_NAME}/antisemitism-results/checkpoints/${CKPT_PREFIX}${SEED}/best_model.pt"
    JOB_NAME="ig-${TAG}-s${SEED}"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/eval_attribution.py \
        --checkpoint ${CKPT} \
        --tag ${TAG} --seed ${SEED} \
        --output results/interpretability/${TAG}_s${SEED}_ig.json \
        --n-examples 500" \
      "${NODE_POOL}"
  done
done

echo ""
echo "IG sweep submitted: ${#MODELS[@]} models × ${#SEEDS[@]} seeds = $((${#MODELS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep ig-'"
echo ""
echo "When all Succeeded, run analysis:"
echo "  python3 analysis/attribution_analysis.py    # writes paper/figures/attribution_shift.pdf"
