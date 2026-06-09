#!/usr/bin/env bash
# Container-side entrypoint called by rcp/submit.sh.
#
# Runai CLI mangles any argument that contains single quotes or multi-line
# strings (splits on quote boundaries → bash -c receives only a prefix → the
# container succeeds silently without running the real command).
#
# To sidestep that entirely, submit.sh calls:
#   bash <ABS_PATH>/rcp/_runner.sh <python command args...>
# which lets runai pass plain word-tokens as argv, and this script handles
# setup + exec. No quoting games in submit.sh.
#
# This script is username-agnostic: it cd's to its own parent of `rcp/`,
# so it works for any team member regardless of where they cloned the repo
# inside their home PVC.
#
# $@ = the python command (e.g. python3 scripts/train.py --config ... --no-wandb)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[_runner] CWD      : $(pwd)"
echo "[_runner] uid/user : $(id -u) ($(id -un 2>/dev/null || echo unknown))"
echo "[_runner] GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "[_runner] results/ : $(readlink results 2>/dev/null || echo 'NOT A SYMLINK — run rcp/setup_home.sh on jumphost')"
echo "[_runner] python   : $(which python3)  ($(python3 --version))"
echo "[_runner] command  : $*"
echo "[_runner] starting..."
echo ""

exec "$@"
