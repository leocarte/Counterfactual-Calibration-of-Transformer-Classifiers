#!/usr/bin/env bash
# Counterfactual hard-flip-rate (CFR_hard) sweep for the three prompted LLMs.
#
# Runs scripts/run_llm_pairs.py on BOTH sides of the shared swap manifest, one
# job per model, on RCP via Run:ai. This fills the dashed LLM CFR cells of the
# main results table (Table 1) with an actual flip-rate measurement.
#
# PREREQUISITE: the shared 2,117-pair manifest must already exist on the group
# scratch PVC. Build it once on the jumphost (CPU-only, ~1 min):
#   python3 analysis/direction1/01_build_swap_manifest.py
# It lands at  $SCRATCH/cfr_cache/_shared/test_swaps_manifest.parquet  and is
# read automatically by run_llm_pairs.py (override with --manifest-path).
#
# Usage:
#   bash rcp/submit_llm_cfr.sh                  # all 3 models, zero-shot
#   bash rcp/submit_llm_cfr.sh llama            # one model, zero-shot
#   bash rcp/submit_llm_cfr.sh qwen guided_cot  # one model, override strategy
#   STRATEGY=few_shot bash rcp/submit_llm_cfr.sh # all 3, override strategy
#   NODE_POOL=h100 bash rcp/submit_llm_cfr.sh    # override GPU node pool (default a100-40g)
#
# Zero-shot is the default and the right choice for a CFR sweep: Guided-CoT
# collapses parser coverage (Mistral CoT ~10% valid), which would compute the
# flip rate over a tiny, biased subset. run_llm_pairs.py reports coverage and
# brackets CFR_hard on the full denominator regardless.
#
# 4-bit 7B/8B models fit comfortably on a100-40g. The sweep prompts both sides
# (~2.7k unique generations/model after greedy dedup); ETA ~35-55 min/model
# zero-shot. run_llm_pairs.py caches every generation and RESUMES on restart,
# so a pre-empted job loses at most one chunk.

set -euo pipefail

STRATEGY="${STRATEGY:-zero_shot}"
NODE_POOL="${NODE_POOL:-default}"   # default pool; override e.g. NODE_POOL=a100-40g bash rcp/submit_llm_cfr.sh
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"   # zero-shot label is short; caps the CoT-config 300

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"

# Match the standard LLM jobs' container entrypoint (rcp/submit_llm.sh).
export RUNNER="/home/${RCP_USERNAME}/antisemitism-detection/rcp/_runner_llm.sh"

# Positional args: a known family selects one model; anything else is a strategy.
FAMILIES=("llama" "mistral" "qwen")
case "${1:-}" in
  llama|mistral|qwen)
    FAMILIES=("${1}")
    [[ -n "${2:-}" ]] && STRATEGY="${2}"
    ;;
  "" ) ;;
  * ) STRATEGY="${1}" ;;   # e.g. `submit_llm_cfr.sh few_shot` -> all 3 models
esac

config_prefix() {
  case "${1}" in
    llama)   echo "llm" ;;
    mistral) echo "mistral" ;;
    qwen)    echo "qwen" ;;
  esac
}

for fam in "${FAMILIES[@]}"; do
  pref="$(config_prefix "${fam}")"
  # The guided-CoT config supplies model_name/quantization; --prompt-strategy
  # overrides the template, so the same config serves any strategy.
  bash "${HERE}/submit.sh" "llmcfr-${fam}-${STRATEGY}" \
    "python3 scripts/run_llm_pairs.py --model-config configs/${pref}_guidedcot.yaml --prompt-strategy ${STRATEGY} --max-new-tokens ${MAX_NEW_TOKENS}" \
    "${NODE_POOL}"
done

echo ""
echo "LLM CFR jobs submitted (strategy=${STRATEGY}, pool=${NODE_POOL}, models=${FAMILIES[*]})."
echo "Stage the manifest first if you have not: python3 analysis/direction1/01_build_swap_manifest.py"
echo "Watch: watch -n 30 'runai list -p course-ee-559-\$(whoami) | grep llmcfr-'"
