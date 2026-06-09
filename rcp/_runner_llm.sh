#!/usr/bin/env bash
# Container-side entrypoint for LLM jobs (D1/D2/D3).
# Same as _runner.sh but installs vllm first — we skip vllm in the Docker image
# because it's a ~10 GB single layer that breaks push on most networks.
#
# Username-agnostic: cd's to its own parent of `rcp/`, works for any team member.
#
# $@ = the python command (e.g. python3 scripts/run_llm.py --config ...)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[_runner_llm] CWD      : $(pwd)"
echo "[_runner_llm] uid/user : $(id -u) ($(id -un 2>/dev/null || echo unknown))"
echo "[_runner_llm] GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "[_runner_llm] python   : $(which python3)  ($(python3 --version))"
export PATH="${HOME}/.local/bin:${PATH}"
PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYUSERBASE="${HOME}/.local"
PYUSERSITE="${PYUSERBASE}/lib/python${PYVER}/site-packages"
mkdir -p "${PYUSERSITE}"
export PYTHONUSERBASE="${PYUSERBASE}"
export PYTHONPATH="${PYUSERSITE}${PYTHONPATH:+:${PYTHONPATH}}"
echo "[_runner_llm] installing vllm (one-off, ~3-5 min)..."
python3 -m pip install --target "${PYUSERSITE}" 'vllm>=0.3.0'
echo "[_runner_llm] vllm installed"
echo "[_runner_llm] command  : $*"
echo "[_runner_llm] starting..."
echo ""

exec "$@"
