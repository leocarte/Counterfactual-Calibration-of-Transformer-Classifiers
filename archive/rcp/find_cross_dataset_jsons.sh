#!/usr/bin/env bash
# Run this from the jumphost (after `ssh <gaspar>@jumphost.rcp.epfl.ch`).
# It locates every `full_evaluation.json` file under your scratch volume that
# matches the cross-dataset pattern, so you can verify what's actually been
# evaluated before pulling.
#
# Usage:
#   ssh <gaspar>@jumphost.rcp.epfl.ch
#   bash find_cross_dataset_jsons.sh        # or paste the body below

set -euo pipefail

SCRATCH="/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/${USER}/antisemitism-results"

echo "Scratch root: ${SCRATCH}"
echo

if [[ ! -d "${SCRATCH}" ]]; then
  echo "ERROR: scratch root not found. Check the path or ${USER}." >&2
  exit 1
fi

# Look in the two most plausible parents.
echo "===  Candidate parents under SCRATCH  ==="
ls -1 "${SCRATCH}" 2>/dev/null | sed 's/^/  /'
echo

# Find every cross_dataset_*/full_evaluation.json
echo "===  full_evaluation.json files matching cross_dataset_* pattern  ==="
find "${SCRATCH}" \
  -type f -name 'full_evaluation.json' -path '*cross_dataset*' \
  2>/dev/null | sort | sed "s|${SCRATCH}/|  |"
echo

# Tally per (run_name) to confirm seed coverage.
echo "===  Per-run seed coverage  ==="
find "${SCRATCH}" -type f -name 'full_evaluation.json' -path '*cross_dataset*' 2>/dev/null \
  | sed -E "s|.*/(cross_dataset_[^/]+_seed[0-9]+)/.*|\1|" \
  | sed -E "s|_seed[0-9]+||" \
  | sort | uniq -c \
  | awk '{printf "  %-60s %d seeds\n", $2, $1}'
echo

echo "Done.  If everything looks right, run scp/rsync next (see rsync_cross_dataset.sh)."
