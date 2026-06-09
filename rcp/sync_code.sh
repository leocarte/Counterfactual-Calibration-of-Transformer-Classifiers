#!/usr/bin/env bash
# Push local code to /home/case/antisemitism-detection on the RCP home PVC.
#
# Jobs mount the home PVC into the container, so keeping /home/case in sync
# with your laptop lets you iterate without rebuilding the Docker image
# (rebuild only when requirements.txt changes).
#
# Run from the repo root: `bash rcp/sync_code.sh`
#
# Prereqs:
#   - Connected to EPFL WiFi or VPN
#   - SSH key set up for jumphost.rcp.epfl.ch
#
# This uses rsync over SSH via the jumphost.

set -euo pipefail

# Parametrised on $USER so any teammate can run without overrides.
REMOTE_USER="${REMOTE_USER:-$(whoami)}"
REMOTE="${REMOTE:-${REMOTE_USER}@jumphost.rcp.epfl.ch}"
REMOTE_DIR="${REMOTE_DIR:-/home/${REMOTE_USER}/antisemitism-detection}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing ${LOCAL_DIR} → ${REMOTE}:${REMOTE_DIR}"

# Exclude: venv caches, artifacts, git metadata, local-only dirs.
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pyc' \
  --exclude 'results/' \
  --exclude 'data/processed/' \
  --exclude 'wandb/' \
  --exclude '.venv/' \
  --exclude 'rcp/.docker_build_cache/' \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

echo "Done. Remote code is at ${REMOTE_DIR}."
echo "Note: data/raw/ IS synced — make sure GoldStandard2024.csv is local before first sync."
