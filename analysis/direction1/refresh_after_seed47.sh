#!/usr/bin/env bash
# Fold CCI DeBERTa-v3-large seed 47 into the Direction-1 rollups.
#
# Until seed 47 landed, that cell was the paper's only n=5. Closing it changes:
# the Table 1 CCI-DeBERTa-large row, the CFR_hard reduction headline (55% -> 58%),
# the n=5 sentence, and the paired bootstrap, which fell back to the five-seed
# intersection. It also lets the exact sign test reach significance on that cell:
# with n=5 its floor is 2*2^-5 = 0.0625, above alpha even when all seeds agree.
#
# NOTE: analysis/direction1/_utils.py:_SEEDS_OVERRIDE must be empty, or every
# step below silently keeps using five seeds for this cell. That override is what
# made the first attempt a no-op.
#
# Step 1 is the only GPU work (one checkpoint). Steps 2-6 are CPU but still need
# the container's pandas/torch, so the whole chain runs as one job.
#
# Order matters: 06 consumes the outputs of 03, 04 and 05; 07 consumes all of them.
#
# 02_forward_pass is idempotent (compares checkpoint sha256 + swap_seed against the
# cache), so re-running the whole script is safe.
#
# Submit with:
#   bash rcp/submit.sh d1-refresh-s47 "bash analysis/direction1/refresh_after_seed47.sh" a100-40g

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXP="cci_v2_fixed_js_deberta_v3_large"
CKPT="results/checkpoints/${EXP}_seed47/best_model.pt"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint at ${CKPT}" >&2
  echo "        Either seed 47 did not finish, or results/ is not the scratch symlink." >&2
  exit 1
fi

# Guard against the failure mode that wasted the first run: the seed registry,
# not the checkpoint, decides which seeds the rollups see. Test the DICT, not the
# experiment name -- that name also appears in MULTIARCH_METHODS and the display
# maps, so a plain grep is a guaranteed false positive.
if ! python3 -c 'import re,io,sys; s=io.open("analysis/direction1/_utils.py",encoding="utf-8").read(); m=re.search(r"_SEEDS_OVERRIDE[^=]*=\s*\{(.*?)\}", s, re.S); sys.exit(0 if (m and not m.group(1).strip()) else 1)'; then
  echo "[ERROR] _SEEDS_OVERRIDE is not empty; this cell would stay pinned to 5 seeds." >&2
  echo "        Empty the dict in analysis/direction1/_utils.py before running." >&2
  exit 1
fi

echo "[refresh] checkpoint OK, seed override clear."

echo ""; echo "=== 1/6  forward pass (GPU) — seed 47 only ==="
python3 analysis/direction1/02_forward_pass.py --experiment "${EXP}" --seed 47

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
echo "[refresh] done. Every rollup should now report n=6 for ${EXP}:"
echo "  grep -i deberta_v3_large analysis/direction1/results/cfr_summary.csv"
echo "  grep -i deberta_v3_large analysis/direction1/results/fped_fned_summary.csv"
echo "  grep -i deberta_v3_large analysis/direction1/results/group_ece_summary.csv"
