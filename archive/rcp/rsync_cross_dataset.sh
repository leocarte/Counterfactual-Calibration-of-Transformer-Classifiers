#!/usr/bin/env bash
# Pull every cross_dataset_*/full_evaluation.json from your RCP scratch volume
# down to the local repo. Run this from your LAPTOP (not the jumphost).
#
# Prereqs:
#   - You can `ssh <gaspar>@jumphost.rcp.epfl.ch` without password (SSH key set up)
#   - You first ran `bash rcp/find_cross_dataset_jsons.sh` on the jumphost and
#     confirmed the files exist
#
# Usage:
#   cd /path/to/DeepLearning
#   GASPAR=case bash rcp/rsync_cross_dataset.sh
#   (replace GASPAR=case with your own gaspar username if different)

set -euo pipefail

GASPAR="${GASPAR:-case}"                                # override with GASPAR=yourname
HOST="jumphost.rcp.epfl.ch"
SCRATCH="/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/${GASPAR}/antisemitism-results"
DEST="results/metrics_from_rcp"

mkdir -p "${DEST}"

echo "Pulling cross_dataset JSONs from ${GASPAR}@${HOST}:${SCRATCH}"
echo "Destination: ${DEST}/"
echo

# --include / --exclude filter: descend into directories, copy only JSONs.
# This avoids pulling parquets/checkpoints/etc.
rsync -avz --prune-empty-dirs \
  --include='cross_dataset_*/' \
  --include='cross_dataset_*/*/' \
  --include='cross_dataset_*/**/full_evaluation.json' \
  --include='cross_dataset_*/**/test_metrics.json' \
  --exclude='*' \
  "${GASPAR}@${HOST}:${SCRATCH}/" \
  "${DEST}/"

echo
echo "Verifying what landed locally:"
find "${DEST}" -type f -name 'full_evaluation.json' -path '*cross_dataset*' | sort | sed 's|^|  |'
echo
echo "Done. Next: python scripts/aggregate_cross_dataset.py"
