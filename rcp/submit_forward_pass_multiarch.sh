#!/usr/bin/env bash
# CFR-evaluation forward-pass — multi-architecture extension.
#
# Same pipeline as submit_forward_pass.sh but for the 4 multi-arch cells
# trained on 2026-05-14:
#   * model_c_deberta_v3_large            (seeds 42-47, n=6)
#   * model_c_roberta_large               (seeds 42-47, n=6, with mask_token fix)
#   * cci_v2_fixed_js_deberta_v3_large    (seeds 42-46, n=5 — s47 dropped, slow on a100)
#   * cci_v2_fixed_js_roberta_large       (seeds 42-47, n=6)
#
# Total: 23 jobs. Output goes to /scratch/case/antisemitism-results/cfr_cache/<exp>_seed<N>/
# alongside the existing 54 base+sweep caches. The downstream scripts
# (03 CFR, 04 FPED/FNED, 05 group-ECE) read these caches by experiment name
# and automatically pick up multi-arch results.
#
# Pre-requisite: shared swap manifest at
#   /scratch/case/antisemitism-results/cfr_cache/_shared/test_swaps_manifest.parquet
# (same one used by the base sweep — built by 01_build_swap_manifest.py).
#
# Usage:
#   bash rcp/submit_forward_pass_multiarch.sh             # full 23
#   bash rcp/submit_forward_pass_multiarch.sh smoke       # 1 job
#   SEEDS="42" bash rcp/submit_forward_pass_multiarch.sh  # restrict
#
# Override node pool with a second arg (default: ``default``).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-default}"

# Per-experiment seed lists. cci_v2_fixed_js_deberta_v3_large has 5 seeds
# (s47 was dropped during training); the others have 6.
declare -A METHOD_SEEDS=(
  [model_c_deberta_v3_large]="42 43 44 45 46 47"
  [model_c_roberta_large]="42 43 44 45 46 47"
  [cci_v2_fixed_js_deberta_v3_large]="42 43 44 45 46"
  [cci_v2_fixed_js_roberta_large]="42 43 44 45 46 47"
)

# Optional global seed restriction
SEEDS_OVERRIDE="${SEEDS:-}"

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

submit_one () {
  local exp="$1"
  local seed="$2"
  local job_name
  job_name="d1-fwd-ma-${exp//_/-}-s${seed}"
  job_name="${job_name//[^a-z0-9-]/-}"
  job_name="${job_name:0:50}"
  bash "${REPO_ROOT}/rcp/submit.sh" "${job_name}" \
    "${PY_BASE} --experiment ${exp} --seed ${seed}" \
    "${NODE_POOL}"
}

if [[ "${MODE}" == "smoke" ]]; then
  EXP="cci_v2_fixed_js_deberta_v3_large"
  SEED=42
  echo "Submitting multi-arch forward-pass smoke: ${EXP} seed ${SEED}"
  submit_one "${EXP}" "${SEED}"
  echo ""
  echo "Tail logs: runai logs d1-fwd-ma-${EXP//_/-}-s${SEED} -p course-ee-559-\$(whoami)"
  exit 0
fi

n_jobs=0
for EXP in "${!METHOD_SEEDS[@]}"; do
  read -ra SEEDS_FOR_EXP <<< "${METHOD_SEEDS[$EXP]}"
  for SEED in "${SEEDS_FOR_EXP[@]}"; do
    # Apply global SEEDS restriction if set
    if [[ -n "${SEEDS_OVERRIDE}" ]] && ! [[ " ${SEEDS_OVERRIDE} " =~ " ${SEED} " ]]; then
      continue
    fi
    submit_one "${EXP}" "${SEED}"
    n_jobs=$((n_jobs + 1))
  done
done

echo ""
echo "Multi-arch forward-pass submitted: ${n_jobs} jobs on node pool '${NODE_POOL}'."
echo "Monitor with:"
echo "  watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep d1-fwd-ma-'"
echo ""
echo "When all Succeeded, run downstream scripts on the jumphost:"
echo "  python3 analysis/03_compute_cfr.py"
echo "  python3 analysis/04_compute_fped_fned.py"
echo "  python3 analysis/05_compute_group_ece.py"
