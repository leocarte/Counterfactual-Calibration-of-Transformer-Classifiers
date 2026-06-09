#!/usr/bin/env bash
# Submit a single job to EPFL RCP via Run:ai.
#
# Usage:
#   bash rcp/submit.sh <job_name> "<python_cmd>" [node_pool]
#
# Examples:
#   bash rcp/submit.sh smoke-roberta \
#     "python3 scripts/train.py --config configs/roberta_smoke.yaml --no-wandb"
#   bash rcp/submit.sh llm-d1 \
#     "python3 scripts/run_llm.py --config configs/llm_zeroshot.yaml" a100-80g
#
# Prereqs (one-time per team member):
#   - `runai login` (follow URL, paste token)
#   - `runai config project course-ee-559-${USER}`
#   - Code at /home/${USER}/antisemitism-detection/ on the home PVC
#   - Raw data at /home/${USER}/antisemitism-detection/data/raw/GoldStandard2024.csv
#   - Image already on registry (built once by anyone in the team — public Harbor
#     project, all members can pull): registry.rcp.epfl.ch/ee-559-case/antisemitism:latest
#
# Auto-detected per user:
#   RCP_USERNAME    (default: $(whoami))   — your gaspar username
#   RCP_UID         (default: $(id -u))    — your numeric UID
#
# Overrides via env vars:
#   IMAGE           (default: registry.rcp.epfl.ch/ee-559-case/antisemitism:latest)
#   GPU_COUNT       (default: 1)
#   EXTRA_ARGS      (default: empty) — raw args passed through to `runai submit`
#   RUNNER          (default: /home/${RCP_USERNAME}/antisemitism-detection/rcp/_runner.sh)
#   RUNAI_PROJECT   (default: course-ee-559-${RCP_USERNAME})
#   AUTO_DELETE     (default: 168h)        — clear succeeded jobs after N hours
#   BACKOFF_LIMIT   (default: 1)           — pod-create retry budget on failure

set -euo pipefail

JOB_NAME="${1:?Usage: bash rcp/submit.sh <job_name> \"<python_cmd>\" [node_pool]}"
PY_CMD="${2:?Usage: bash rcp/submit.sh <job_name> \"<python_cmd>\" [node_pool]}"
NODE_POOL="${3:-a100-40g}"

RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"
RCP_UID="${RCP_UID:-$(id -u)}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-case/antisemitism:latest}"
GPU_COUNT="${GPU_COUNT:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUNNER="${RUNNER:-/home/${RCP_USERNAME}/antisemitism-detection/rcp/_runner.sh}"
# Run:ai project — explicit, so we are not at the mercy of a stale `runai
# config project` default. Each EE-559 student has their own project named
# course-ee-559-<gaspar>; multi-account users (e.g. SFI master + EE-559
# course) need this pin so jobs don't accidentally go to a sibling project.
RUNAI_PROJECT="${RUNAI_PROJECT:-course-ee-559-${RCP_USERNAME}}"
# Auto-cleanup of finished jobs (G1) — runai removes the job 168h (1 week) after
# completion to keep `runai list` uncluttered. Override with AUTO_DELETE=24h etc.
AUTO_DELETE="${AUTO_DELETE:-168h}"
# Backoff (G2) — by default Run:ai retries failed pod-creation up to 6x before
# marking the job Failed. Cap at 1 so a bad commit fails fast instead of
# burning through retries.
BACKOFF_LIMIT="${BACKOFF_LIMIT:-1}"
# NB: do NOT set --working-dir on `runai submit`; it causes a permission-denied
# STARTERROR ("error during container init: mkdir ...: permission denied").
# runc applies --working-dir during container init, BEFORE the home PVC
# bind-mount is usable by the --run-as-uid user, so it tries (and fails) to
# mkdir the path on the bare image filesystem. _runner.sh cds explicitly
# inside the container after mounts are in place; that is the only correct
# place for the cd.

# G4 — fail fast if the user is not logged in to runai. Saves time vs the
# obscure error a botched `runai submit` produces in that state.
if ! runai whoami >/dev/null 2>&1; then
  echo "[ERROR] You are not logged in to runai." >&2
  echo "        Run: runai login && runai config project ${RUNAI_PROJECT}" >&2
  exit 1
fi

# G5 — hard guard against accidental cross-project submission. Some team
# members have multiple Run:ai projects on this cluster (e.g. an SFI master
# project alongside the EE-559 course project) and a stale `runai config
# project sfi-...` or a mis-set RUNAI_PROJECT env var could otherwise
# silently route course jobs (and the resulting GPU spend) to the wrong
# project. EE-559 mini-project compute MUST live under course-ee-559-*.
if [[ "${RUNAI_PROJECT}" != course-ee-559-* ]]; then
  echo "[ERROR] Refusing to submit: RUNAI_PROJECT='${RUNAI_PROJECT}' is not an" >&2
  echo "        EE-559 course project. Expected a name like 'course-ee-559-<gaspar>'." >&2
  echo "        Unset RUNAI_PROJECT or export RUNAI_PROJECT=course-ee-559-${RCP_USERNAME} and retry." >&2
  exit 1
fi

# G6 — also pin the runai CLI default to this project, so any ad-hoc
# `runai list`, `runai logs`, `runai delete job` etc. that the user runs
# from their shell hits the EE-559 project rather than whatever happened to
# be the most-recent `runai config project` target (e.g. an SFI project).
#
# Unconditional + `|| true`: this is belt-and-suspenders, NOT the real
# protection (G5's refuse-to-submit + the explicit --project flag on
# `runai submit` below already pin where the job actually lands). If the
# CLI version doesn't accept this command form, we don't want to fail the
# whole submission — silently fall back to the explicit flag.
#
# Always set the project rather than probe-and-compare: `runai config project`
# with no argument exits non-zero on CLIs that require an argument, which under
# `set -euo pipefail` would abort the submit. Setting unconditionally avoids it.
runai config project "${RUNAI_PROJECT}" >/dev/null 2>&1 || true

# Forward HF_TOKEN if set (required for gated Llama/Mistral checkpoints).
HF_TOKEN_ARG=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG="--environment HF_TOKEN=${HF_TOKEN}"
  echo "  HF_TOKEN    : forwarded (length=${#HF_TOKEN})"
fi

# Run:ai requires lowercase + hyphenated job names, max ~50 chars. Normalize.
JOB_NAME="$(echo "${JOB_NAME}" | tr '[:upper:]_' '[:lower:]-' | cut -c 1-50)"

echo "Submitting: ${JOB_NAME}"
echo "  user        : ${RCP_USERNAME} (uid=${RCP_UID})"
echo "  project     : ${RUNAI_PROJECT}"
echo "  image       : ${IMAGE}"
echo "  node pool   : ${NODE_POOL}"
echo "  gpu count   : ${GPU_COUNT}"
echo "  backoff     : ${BACKOFF_LIMIT}"
echo "  auto-delete : ${AUTO_DELETE}"
echo "  command     : ${PY_CMD}"
[[ -n "${EXTRA_ARGS}" ]] && echo "  extra args  : ${EXTRA_ARGS}"

# The container entrypoint delegates to rcp/_runner.sh on the home PVC.
# That script does cd + diagnostics + exec of the python command.
#
# Why a separate script: runai CLI mangles args containing single quotes,
# newlines, or shell metacharacters (splits the string into multiple argv
# entries → bash -c receives only a prefix → container succeeds silently).
# Passing a script path + plain word tokens sidesteps all of that.

runai submit \
  --name "${JOB_NAME}" \
  --project "${RUNAI_PROJECT}" \
  --run-as-uid "${RCP_UID}" \
  --image "${IMAGE}" \
  --gpu "${GPU_COUNT}" \
  --node-pools "${NODE_POOL}" \
  --backoff-limit "${BACKOFF_LIMIT}" \
  --auto-deletion-time-after-completion "${AUTO_DELETE}" \
  --existing-pvc claimname=home,path=/home/${RCP_USERNAME} \
  --existing-pvc claimname=course-ee-559-scratch-g39,path=/scratch \
  ${EXTRA_ARGS} \
  ${HF_TOKEN_ARG} \
  --command -- bash ${RUNNER} ${PY_CMD}
# NOTE: course-ee-559-shared-ro and course-ee-559-shared-rw PVCs are
# intentionally NOT mounted. Reasons: (i) we never read from /shared-ro —
# all our datasets come from Zenodo / HuggingFace at runtime; (ii) /shared-rw
# is course-wide writable, so a misuse from anywhere else could clobber data
# we depend on, and we'd rather not even have it visible. If a future task
# needs course-shared resources, add them back to the runai submit block.

echo ""
echo "Tail logs with : runai logs ${JOB_NAME} -p ${RUNAI_PROJECT}"
echo "Describe       : runai describe job ${JOB_NAME} -p ${RUNAI_PROJECT}"
echo "Kill           : runai delete job ${JOB_NAME} -p ${RUNAI_PROJECT}"
