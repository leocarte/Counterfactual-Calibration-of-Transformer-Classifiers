#!/usr/bin/env bash
# Forward-pass sweep for the semi-supervised training cells.
#
# After rcp/submit_semisup.sh produced checkpoints
#   checkpoints/<base>_lf<frac>_seed<N>/best_model.pt
# this fans out GPU forward-pass jobs writing
#   cfr_cache/<base>_lf<frac>_seed<N>/{test_originals,test_swap_logits,manifest}.parquet
# so analysis/direction1/12_semisup.py can compute F1+/CFR vs label fraction.
#
# Usage:
#   bash analysis/direction1_rcp/submit_forward_semisup.sh
#   bash analysis/direction1_rcp/submit_forward_semisup.sh smoke
#   FRACS="0.1 0.25" SEEDS="42 43" bash analysis/direction1_rcp/submit_forward_semisup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-a100-40g}"
FRACS=(${FRACS:-0.1 0.25 0.5})
SEEDS=(${SEEDS:-42 43 44})
BASES=(model_c_deberta_focal_mask cda_deberta_base cci_v2_fixed_js)

PY_BASE="python3 analysis/direction1/02_forward_pass.py"

submit_one () {
  local exp="$1" seed="$2"
  local job
  job="d1-fwd-${exp//_/-}-s${seed}"
  job="${job//[^a-z0-9-]/-}"
  job="${job:0:50}"
  if ! bash "${REPO_ROOT}/rcp/submit.sh" "${job}" \
       "${PY_BASE} --experiment ${exp} --seed ${seed}" "${NODE_POOL}"; then
    echo "[warn] forward submit failed for ${job} (already exists?), continuing." >&2
  fi
}

if [[ "${MODE}" == "smoke" ]]; then
  submit_one "cci_v2_fixed_js_lf0.1" 42
  exit 0
fi

n=0
for SEED in "${SEEDS[@]}"; do
  for FRAC in "${FRACS[@]}"; do
    for BASE in "${BASES[@]}"; do
      submit_one "${BASE}_lf${FRAC}" "${SEED}"
      n=$((n + 1))
    done
  done
done

echo ""
echo "Semi-sup forward-pass sweep submitted: ${n} jobs."
echo "When all Succeeded, compute the curves on the jumphost:"
echo "  python3 analysis/direction1/12_semisup.py"
