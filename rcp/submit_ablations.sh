#!/usr/bin/env bash
# Launch the full 4-model transformer sweep on RCP.
#
# Models:
#   A  roberta-base          (precision: bf16)
#   B  deberta-v3-base       (precision: fp32 — disentangled attn overflows bf16)
#   B2 hateBERT              (precision: bf16)
#   C  deberta + focal+mask  (precision: fp32, 7 epochs — novel contribution)
#
# Usage: bash rcp/submit_ablations.sh [node_pool]
#   Default: a100-40g (only pool actually available to course-ee-559-case on RCP
#   as of 2026-04-23; 40 GB is plenty for DeBERTa fp32 at batch=4, grad_accum=4).
#
# Each job: ~45-90 min on A100-40G. All four run in parallel on separate GPUs.

set -euo pipefail

NODE_POOL="${1:-a100-40g}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${HERE}/submit.sh" "model-a-roberta" \
  "python3 scripts/train.py --config configs/roberta.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "model-b-deberta" \
  "python3 scripts/train.py --config configs/deberta.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "model-b2-hatebert" \
  "python3 scripts/train.py --config configs/hatebert.yaml --no-wandb" \
  "${NODE_POOL}"

bash "${HERE}/submit.sh" "model-c-deberta-focal-mask" \
  "python3 scripts/train.py --config configs/deberta_focal_mask.yaml --no-wandb" \
  "${NODE_POOL}"

echo ""
echo "Full 4-model sweep submitted. ETA ~90 min wall-clock (runs in parallel)."
