#!/usr/bin/env bash
# Launch LLM inference jobs (Models D1/D2/D3) on RCP.
#
# Models:
#   D1  Instruct LLM — zero-shot
#   D2  Instruct LLM — few-shot (5 examples/class)
#   D3  Instruct LLM — guided chain-of-thought (Patel et al. EMNLP 2025)
#
# Usage:
#   bash rcp/submit_llm.sh                 # full 1060-sample test set (llama, all 3 prompts)
#   bash rcp/submit_llm.sh smoke           # 20-sample smoke sweep (llama)
#   bash rcp/submit_llm.sh mistral         # full sweep with Mistral-7B-Instruct
#   bash rcp/submit_llm.sh qwen smoke      # smoke sweep with Qwen-2.5-7B-Instruct
#
# 4-bit-quantized 7B/8B instruct models need ~5-8 GB VRAM + headroom for kv-cache;
# a100-40g (the pool actually available to course-ee-559-case) is more than enough.
# Full test run ETA ~30-60 min per model.
#
# NOTE: vllm is NOT baked into the Docker image (too big — ~10 GB single layer
# breaks most push networks). We install it here at job startup instead.
# Adds ~3-5 min of setup per job but unblocks the whole workflow.

set -euo pipefail

MODEL_FAMILY="llama"
MODE="full"

if [[ "${1:-}" == "smoke" || "${1:-}" == "full" ]]; then
  MODE="${1}"
elif [[ -n "${1:-}" ]]; then
  MODEL_FAMILY="${1}"
fi

if [[ "${2:-}" == "smoke" || "${2:-}" == "full" ]]; then
  MODE="${2}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"

# Use the LLM-specific runner, which does `pip install vllm` before exec'ing
# the python command. (vllm is not in the Docker image — single 10 GB layer
# breaks push; installing inside the container is the workaround.)
export RUNNER="/home/${RCP_USERNAME}/antisemitism-detection/rcp/_runner_llm.sh"

case "${MODEL_FAMILY}" in
  llama)
    CONFIG_PREFIX="llm"
    JOB_PREFIX="llama"
    ;;
  mistral)
    CONFIG_PREFIX="mistral"
    JOB_PREFIX="mistral"
    ;;
  qwen)
    CONFIG_PREFIX="qwen"
    JOB_PREFIX="qwen"
    ;;
  *)
    echo "[ERROR] Unknown LLM family: ${MODEL_FAMILY}" >&2
    echo "        Expected one of: llama, mistral, qwen" >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "smoke" ]]; then
  bash "${HERE}/submit.sh" "smoke-${JOB_PREFIX}-d1-zero" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_zeroshot_smoke.yaml" a100-40g
  bash "${HERE}/submit.sh" "smoke-${JOB_PREFIX}-d2-few" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_fewshot_smoke.yaml" a100-40g
  bash "${HERE}/submit.sh" "smoke-${JOB_PREFIX}-d3-cot" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_guidedcot_smoke.yaml" a100-40g
else
  bash "${HERE}/submit.sh" "model-${JOB_PREFIX}-d1-zero" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_zeroshot.yaml" a100-40g
  bash "${HERE}/submit.sh" "model-${JOB_PREFIX}-d2-few" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_fewshot.yaml" a100-40g
  bash "${HERE}/submit.sh" "model-${JOB_PREFIX}-d3-cot" \
    "python3 scripts/run_llm.py --config configs/${CONFIG_PREFIX}_guidedcot.yaml" a100-40g
fi

echo ""
echo "LLM jobs submitted (family=${MODEL_FAMILY}, mode=${MODE}). Watch with: watch -n 5 runai list"
