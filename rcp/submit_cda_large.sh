#!/usr/bin/env bash
# CDA at LARGE scale — closes the one gap the WOAH reviewers can point at.
#
# Reviewer #1: "CDA achieves a lower CFR_hard than CCI in every comparison."
# True, but there is exactly ONE matched comparison in the paper (DeBERTa-base).
# CDA was never run on the large backbones, so we cannot say whether its edge
# survives at scale — which is precisely where CCI's "no doubled training set"
# argument stops being bookkeeping and starts costing real GPU hours.
#
# This sweep runs CDA on both large backbones with hyperparameters copied
# verbatim from model_c_*_large.yaml, so CDA-vs-C and CDA-vs-CCI stay matched
# at large scale exactly as they are at base scale.
#
#   configs/cda_deberta_v3_large.yaml   (440M, same family as the CCI pivot)
#   configs/cda_roberta_large.yaml      (355M, cross-family)
#
# 2 cells x 6 seeds = 12 jobs. CDA doubles the train split but does NOT expand
# counterfactuals per step (that is CCI), so each step is lighter than a CCI
# step while there are ~2x as many. Budget ~1-1.5 h per run on A100.
#
# Usage:
#   bash rcp/submit_cda_large.sh smoke          # 2 cells x seed 42 x 1 epoch  <-- RUN THIS FIRST
#   bash rcp/submit_cda_large.sh                # full 12-job sweep
#   SEEDS="42 43" bash rcp/submit_cda_large.sh  # restricted seed list
#   bash rcp/submit_cda_large.sh full a100-80g  # alternate node pool
#
# Seeds {42..47} match every other sweep, so the cells pool with the existing
# table rows.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
# a100-40g, not `default`: submit_ablations.sh and submit_llm.sh both record that
# this is the only pool actually available to course-ee-559-case, and `default`
# races on GPU admission.
NODE_POOL="${2:-a100-40g}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

declare -a CELLS=(
  "cda_deberta_v3_large"
  "cda_roberta_large"
)

# Smoke first. 12 large-model jobs is the wrong place to discover a config typo:
# each real run is ~1-1.5 h, so a bad sweep burns most of a day.
if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting CDA-large smoke: ${#CELLS[@]} cells x seed 42 x 1 epoch..."
  for CFG in "${CELLS[@]}"; do
    JOB_NAME="$(echo "cdal-${CFG//_/-}-smoke" | cut -c 1-50)"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/train.py --config configs/${CFG}_smoke.yaml --no-wandb --seed 42" \
      "${NODE_POOL}"
  done
  echo ""
  echo "Smoke submitted. Verify BOTH reach 'Epoch 1/1' and write predictions, then"
  echo "run the full sweep. Check the cda_augmentation branch actually fired:"
  echo "  runai logs cdal-cda-deberta-v3-large-smoke -p course-ee-559-\$(whoami) | grep -i 'augment\|train size\|expanded'"
  exit 0
fi

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CELLS[@]}"; do
    JOB_NAME="$(echo "cdal-${CFG//_/-}-s${SEED}" | cut -c 1-50)"
    bash "${HERE}/submit.sh" "${JOB_NAME}" \
      "python3 scripts/train.py --config configs/${CFG}.yaml --no-wandb --seed ${SEED}" \
      "${NODE_POOL}"
  done
done

echo ""
echo "CDA-large sweep submitted: ${#CELLS[@]} cells x ${#SEEDS[@]} seeds = $((${#CELLS[@]} * ${#SEEDS[@]})) jobs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep cdal-'"
echo ""
echo "When they finish, refresh the analysis (CPU, from cached probabilities):"
echo "  python3 analysis/direction1/02_forward_pass.py"
echo "  python3 analysis/direction1/03_compute_cfr.py"
echo "  python3 analysis/direction1/05_compute_group_ece.py"
echo "  python3 analysis/direction1/06_run_significance_suite.py"
