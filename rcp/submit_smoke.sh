#!/usr/bin/env bash
# Launch all 4 transformer smokes on RCP (1 epoch each, pipeline validation).
#
# Usage: bash rcp/submit_smoke.sh [node_pool]
#   Default node pool: a100-40g (smoke jobs are tiny, don't need 80G)
#
# After submission, monitor with:
#   runai list
#   runai logs smoke-<model>
#
# Each smoke takes ~5-10 min on A100. All four run in parallel.

set -euo pipefail

NODE_POOL="${1:-a100-40g}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${HERE}/submit.sh" "smoke-roberta" \
  "python3 scripts/train.py --config configs/roberta_smoke.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "smoke-deberta" \
  "python3 scripts/train.py --config configs/deberta_smoke.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "smoke-hatebert" \
  "python3 scripts/train.py --config configs/hatebert_smoke.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "smoke-deberta-focal-mask" \
  "python3 scripts/train.py --config configs/deberta_focal_mask_smoke.yaml --no-wandb" \
  "${NODE_POOL}"

echo ""
echo "All 4 transformer smokes submitted. Watch with: watch -n 5 runai list"
