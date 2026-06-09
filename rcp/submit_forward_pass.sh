#!/usr/bin/env bash
# CFR-evaluation forward-pass sweep — fan out 54 RCP jobs (one per checkpoint).
#
# Each job runs analysis/02_forward_pass.py for ONE
# (experiment, seed) tuple, producing a parquet cache under
#   /scratch/case/antisemitism-results/cfr_cache/<exp>_seed<N>/
#
# 8 sweep methods × 6 seeds + 1 baseline × 6 seeds = 54 jobs total.
#
# Pre-requisite: the shared swap manifest must already exist. Build it once
# on the jumphost with:
#   python3 analysis/01_build_swap_manifest.py
#
# Usage:
#   bash rcp/submit_forward_pass.sh                  # full 54
#   bash rcp/submit_forward_pass.sh smoke            # 1 job
#   SEEDS="42 43" bash rcp/submit_forward_pass.sh    # restrict
#
# Override the node pool by adding it as a second arg (default a100-40g).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-a100-40g}"
SEEDS=(${SEEDS:-42 43 44 45 46 47})

# 8 sweep methods (training-experiments branch) + 1 baseline (Model C).
SWEEP_METHODS=(
  davani_lambda05
  davani_lambda10
  davani_lambda20
  davani_lambda10_nofilt
  cci_v2_fixed_l1
  cci_v2_fixed_js
  cci_v2_learned_l1
  cci_v2_learned_js
)
BASELINE_METHODS=(
  model_c_deberta_focal_mask
)

SHARED_MANIFEST=/scratch/case/antisemitism-results/cfr_cache/_shared/test_swaps_manifest.parquet
SHARED_MANIFEST_HOST=/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/case/antisemitism-results/cfr_cache/_shared/test_swaps_manifest.parquet

if [[ ! -e "${SHARED_MANIFEST_HOST}" ]]; then
  echo "[ERROR] Shared swap manifest not found at:" >&2
  echo "        ${SHARED_MANIFEST_HOST}" >&2
  echo "        Run on the jumphost first:" >&2
  echo "          python3 analysis/01_build_swap_manifest.py" >&2
  exit 1
fi
echo "[ok] Shared swap manifest present: ${SHARED_MANIFEST_HOST}"

PY_BASE="python3 analysis/02_forward_pass.py"

if [[ "${MODE}" == "smoke" ]]; then
  EXP="cci_v2_learned_js"
  SEED=42
  JOB_NAME="d1-fwd-${EXP//_/-}-s${SEED}"
  JOB_NAME="${JOB_NAME//[^a-z0-9-]/-}"
  JOB_NAME="${JOB_NAME:0:50}"
  echo "Submitting smoke: ${JOB_NAME}"
  bash "${REPO_ROOT}/rcp/submit.sh" "${JOB_NAME}" \
    "${PY_BASE} --experiment ${EXP} --seed ${SEED}" \
    "${NODE_POOL}"
  echo ""
  echo "Tail logs: runai logs ${JOB_NAME} -p course-ee-559-\$(whoami)"
  exit 0
fi

submit_one () {
  local exp="$1"
  local seed="$2"
  local job_name
  job_name="d1-fwd-${exp//_/-}-s${seed}"
  job_name="${job_name//[^a-z0-9-]/-}"
  job_name="${job_name:0:50}"
  bash "${REPO_ROOT}/rcp/submit.sh" "${job_name}" \
    "${PY_BASE} --experiment ${exp} --seed ${seed}" \
    "${NODE_POOL}"
}

n_jobs=0
for SEED in "${SEEDS[@]}"; do
  for EXP in "${SWEEP_METHODS[@]}"; do
    submit_one "${EXP}" "${SEED}"
    n_jobs=$((n_jobs + 1))
  done
  for EXP in "${BASELINE_METHODS[@]}"; do
    submit_one "${EXP}" "${SEED}"
    n_jobs=$((n_jobs + 1))
  done
done

echo ""
echo "CFR-evaluation forward-pass sweep submitted: ${n_jobs} jobs."
echo "Monitor with:"
echo "  watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep d1-fwd'"
