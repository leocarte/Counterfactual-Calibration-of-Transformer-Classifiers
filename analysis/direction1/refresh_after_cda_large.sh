#!/usr/bin/env bash
# Fold the 12 CDA-large runs (2 backbones x 6 seeds) into the Direction-1 rollups.
#
# Closes the gap WOAH Reviewer #1 pointed at: "CDA achieves a lower CFR_hard than
# CCI in every comparison" was true, but there was exactly ONE matched comparison
# (DeBERTa-base). CDA had never been run on the large backbones -- which is where
# CCI's efficiency argument stops being bookkeeping, since the expansion measured
# 8,479 -> 25,762 rows (3.04x, identical across all six seeds).
#
# PREREQUISITE: both cells must be registered in analysis/direction1/_utils.py,
# in MULTIARCH_METHODS *and* METHOD_DISPLAY. all_checkpoints() enumerates from the
# first; assert_methods_mapped fails fast without the second. The guard below
# refuses to run otherwise -- the same class of silent no-op that wasted the first
# seed-47 refresh, where the rollups happily reported stale numbers.
#
# Step 1 is GPU (12 checkpoints, a few minutes each). Steps 2-6 are CPU but need
# the container's pandas/torch, so the whole chain runs as one job.
#
# Submit with:
#   bash rcp/submit.sh d1-refresh-cda-large "bash analysis/direction1/refresh_after_cda_large.sh" a100-40g

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPS=("cda_deberta_v3_large" "cda_roberta_large")
SEEDS=(42 43 44 45 46 47)

# --- guard 1: every checkpoint present, before touching any rollup ---
missing=0
for e in "${EXPS[@]}"; do
  for s in "${SEEDS[@]}"; do
    if [[ ! -f "results/checkpoints/${e}_seed${s}/best_model.pt" ]]; then
      echo "[ERROR] missing checkpoint: ${e}_seed${s}" >&2
      missing=$((missing + 1))
    fi
  done
done
if (( missing > 0 )); then
  echo "[ERROR] ${missing} checkpoint(s) missing -- refusing to refresh partial rollups." >&2
  exit 1
fi

# --- guard 2: both cells registered, or every step below silently ignores them ---
for e in "${EXPS[@]}"; do
  if ! grep -q "\"${e}\"" analysis/direction1/_utils.py; then
    echo "[ERROR] ${e} is not registered in analysis/direction1/_utils.py." >&2
    echo "        Add it to MULTIARCH_METHODS and METHOD_DISPLAY first." >&2
    exit 1
  fi
done

echo "[refresh] 12 checkpoints present, both cells registered."

echo ""; echo "=== 1/6  forward pass (GPU) — 12 checkpoints ==="
for e in "${EXPS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "--- ${e} seed ${s} ---"
    python3 analysis/direction1/02_forward_pass.py --experiment "${e}" --seed "${s}"
  done
done

echo ""; echo "=== 2/6  CFR ==="
python3 analysis/direction1/03_compute_cfr.py

echo ""; echo "=== 3/6  FPED / FNED ==="
python3 analysis/direction1/04_compute_fped_fned.py

echo ""; echo "=== 4/6  group ECE ==="
python3 analysis/direction1/05_compute_group_ece.py

echo ""; echo "=== 5/6  significance suite ==="
python3 analysis/direction1/06_run_significance_suite.py

echo ""; echo "=== 6/6  main table ==="
python3 analysis/direction1/07_build_main_table.py

echo ""
echo "[refresh] done. The two new cells should now appear with n=6:"
echo "  grep -iE 'cda_(deberta_v3|roberta)_large' analysis/direction1/results/cfr_summary.csv"
echo "  grep -iE 'cda_(deberta_v3|roberta)_large' analysis/direction1/results/fped_fned_summary.csv"
