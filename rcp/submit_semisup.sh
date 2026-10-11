#!/usr/bin/env bash
# Semi-supervised sweep -- CCI's structurally-unique edge over CDA.
#
# At label fraction f, all three methods get the SAME labeled subset; only CCI
# additionally enforces (label-free) consistency on the unlabeled (1-f) pool's
# swaps. CDA needs a label to copy onto a swap, so it cannot use the unlabeled
# pool; Model C cannot either. Prediction: CCI's F1+/CFR advantage over CDA GROWS
# as f shrinks. Fraction-1.0 baselines = the existing main runs.
#
# 3 methods x N fractions x M seeds. Default 3 x 3 x 6 = 54 training jobs.
#
# Six seeds, not three: the paper's significance machinery is built on n=6, and
# the exact sign test has a floor of 2*2^-n. At n=3 that floor is 0.25, so the
# test cannot reject even when all three seeds agree in direction -- the same
# wall that made the n=5 DeBERTa-large cell untestable. Any result meant for the
# paper therefore needs six. These are all SMALL runs (fractions 0.1/0.25/0.5 of
# an 8,479-row train split), so the extra 27 jobs cost little.
#   model_c : scripts/train.py     configs/deberta_focal_mask.yaml
#   cda     : scripts/train.py     configs/cda_deberta_base.yaml
#   cci     : scripts/train_cci.py configs/cci_v2_fixed_js.yaml
#
# Output experiment dirs: <prefix>_lf<frac>_seed<N>
#
# Usage:
#   bash rcp/submit_semisup.sh                 # full 27-job sweep
#   bash rcp/submit_semisup.sh smoke           # single cci lf0.1 s42 smoke
#   FRACS="0.1 0.25 0.5" SEEDS="42 43 44" bash rcp/submit_semisup.sh   # quick look

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
NODE_POOL="${2:-default}"
FRACS=(${FRACS:-0.1 0.25 0.5})
SEEDS=(${SEEDS:-42 43 44 45 46 47})

run_method () {
  local method="$1" frac="$2" seed="$3"
  local tag="${frac/./}"   # 0.25 -> 025 (Run:ai job names disallow dots)
  local script cfg job
  case "${method}" in
    model_c) script="scripts/train.py";     cfg="configs/deberta_focal_mask.yaml" ;;
    cda)     script="scripts/train.py";     cfg="configs/cda_deberta_base.yaml" ;;
    cci)     script="scripts/train_cci.py"; cfg="configs/cci_v2_fixed_js.yaml" ;;
    *) echo "unknown method ${method}" >&2; return 2 ;;
  esac
  job="te-lf${tag}-${method//_/-}-s${seed}"
  if ! bash "${HERE}/submit.sh" "${job}" \
       "python3 ${script} --config ${cfg} --no-wandb --seed ${seed} --label-fraction ${frac}" \
       "${NODE_POOL}"; then
    echo "[warn] submit failed for ${job} (already exists?), continuing." >&2
  fi
}

if [[ "${MODE}" == "smoke" ]]; then
  echo "Submitting single semi-sup smoke (cci, fraction 0.1, seed 42)..."
  run_method cci 0.1 42
  exit 0
fi

n=0
for SEED in "${SEEDS[@]}"; do
  for FRAC in "${FRACS[@]}"; do
    for METHOD in model_c cda cci; do
      run_method "${METHOD}" "${FRAC}" "${SEED}"
      n=$((n + 1))
    done
  done
done

echo ""
echo "Semi-sup sweep submitted: ${n} jobs (3 methods x ${#FRACS[@]} fracs x ${#SEEDS[@]} seeds)."
echo "Fraction-1.0 baselines = existing main runs."
echo "Monitor: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep te-lf'"
