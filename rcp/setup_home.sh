#!/usr/bin/env bash
# One-time setup on the RCP jumphost for any team member:
#   - create the per-user scratch results dir on the group scratch PVC (g39)
#   - symlink ~/antisemitism-detection/results -> /scratch/<user>/... (resolves
#     correctly INSIDE a job; the symlink looks broken on the jumphost itself).
#
# The scratch PVC has two paths:
#   - Jumphost-visible : /mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/<user>/antisemitism-results
#   - Inside-a-job     : /scratch/<user>/antisemitism-results
# Same underlying storage; different mount points depending on context.
#
# We mkdir at the jumphost-visible path (where we are NOW), but the symlink points
# to the inside-job path because that's what jobs see.
#
# Auto-detected:
#   RCP_USERNAME  (default: $(whoami))  — your gaspar username
#
# Run this ONCE from the jumphost as yourself:
#
#   ssh <yourgaspar>@jumphost.rcp.epfl.ch
#   cd ~/antisemitism-detection
#   bash rcp/setup_home.sh
#
# Re-running is safe — it's idempotent.

set -euo pipefail

RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"

SCRATCH_JUMPHOST="/mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/${RCP_USERNAME}/antisemitism-results"
SCRATCH_JOB="/scratch/${RCP_USERNAME}/antisemitism-results"
HOME_CODE="/home/${RCP_USERNAME}/antisemitism-detection"

if [[ ! -d "${HOME_CODE}" ]]; then
  echo "ERROR: ${HOME_CODE} does not exist. Clone the repo there first:" >&2
  echo "  git clone https://github.com/<your-username>/DeepLearning.git ~/antisemitism-detection" >&2
  exit 1
fi

echo "Setting up for user: ${RCP_USERNAME}"
echo "Creating scratch dirs at jumphost path: ${SCRATCH_JUMPHOST}"
mkdir -p "${SCRATCH_JUMPHOST}"/{checkpoints,predictions,metrics,figures,processed}

cd "${HOME_CODE}"

if [[ -d results && ! -L results ]]; then
  BACKUP="results.local_$(date +%Y%m%d_%H%M%S)"
  echo "results/ is a real directory. Backing it up to ${BACKUP}/ before symlinking."
  mv results "${BACKUP}"
fi

ln -sfn "${SCRATCH_JOB}" results
echo "results/ -> $(readlink results)  (resolves to real data inside jobs)"

mkdir -p data
if [[ -d data/processed && ! -L data/processed ]]; then
  BACKUP="data/processed.local_$(date +%Y%m%d_%H%M%S)"
  echo "data/processed/ is a real directory. Backing it up to ${BACKUP}/."
  mv data/processed "${BACKUP}"
fi
ln -sfn "${SCRATCH_JOB}/processed" data/processed
echo "data/processed -> $(readlink data/processed)"

echo ""
echo "Setup complete for ${RCP_USERNAME}."
echo "  Jumphost sees scratch at : ${SCRATCH_JUMPHOST}"
echo "  Inside jobs              : ${SCRATCH_JOB}"
echo ""
echo "Other team members' results (read-only via group g39):"
ls -d /mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/*/antisemitism-results 2>/dev/null || true
echo ""
echo "Raw data should live at:"
echo "  ${HOME_CODE}/data/raw/GoldStandard2024.csv"
