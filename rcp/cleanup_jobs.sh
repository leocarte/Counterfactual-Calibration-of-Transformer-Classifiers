#!/usr/bin/env bash
# Bulk-delete finished Run:ai jobs to keep `runai list` uncluttered.
#
# Run:ai's --auto-deletion-time-after-completion (set in rcp/submit.sh) will
# eventually clean Succeeded jobs on its own (after 168h by default). This
# wrapper lets you wipe the queue *manually* and *selectively*, e.g. right
# after pulling results: `bash rcp/cleanup_jobs.sh succeeded`.
#
# Usage:
#   bash rcp/cleanup_jobs.sh [filter] [-n|--dry-run]
#
#   filter:
#     succeeded   (default) — delete only Succeeded jobs.
#     failed                — delete only Failed jobs.
#     all                   — delete every job (Succeeded + Failed). Confirms first.
#     <substring>           — delete only jobs whose name contains the substring,
#                             across any status. Confirms first.
#
# Examples:
#   bash rcp/cleanup_jobs.sh                       # nuke Succeeded (default)
#   bash rcp/cleanup_jobs.sh succeeded
#   bash rcp/cleanup_jobs.sh failed                # nuke Failed only
#   bash rcp/cleanup_jobs.sh smoke                 # nuke jobs with "smoke" in name
#   bash rcp/cleanup_jobs.sh succeeded --dry-run   # show what would be deleted

set -euo pipefail

FILTER="${1:-succeeded}"
DRY_RUN="false"
[[ "${2:-}" == "-n" || "${2:-}" == "--dry-run" ]] && DRY_RUN="true"

RCP_USERNAME="${RCP_USERNAME:-$(whoami)}"
RUNAI_PROJECT="${RUNAI_PROJECT:-course-ee-559-${RCP_USERNAME}}"

if ! runai whoami >/dev/null 2>&1; then
  echo "[ERROR] Not logged in to runai. Run: runai login" >&2
  exit 1
fi

# Pull all jobs in the project once.
ALL=$(runai list -p "${RUNAI_PROJECT}" 2>/dev/null | tail -n +3)
if [[ -z "${ALL}" ]]; then
  echo "No jobs in project ${RUNAI_PROJECT}."
  exit 0
fi

case "${FILTER,,}" in
  succeeded)
    JOBS=$(echo "${ALL}" | awk '$2=="Succeeded" {print $1}')
    LABEL="Succeeded"
    ;;
  failed)
    JOBS=$(echo "${ALL}" | awk '$2=="Failed" {print $1}')
    LABEL="Failed"
    ;;
  all)
    JOBS=$(echo "${ALL}" | awk '$2=="Succeeded" || $2=="Failed" {print $1}')
    LABEL="ALL Succeeded+Failed"
    ;;
  *)
    JOBS=$(echo "${ALL}" | awk -v f="${FILTER}" '$1 ~ f {print $1}')
    LABEL="matching '${FILTER}'"
    ;;
esac

if [[ -z "${JOBS}" ]]; then
  echo "No ${LABEL} jobs to delete in project ${RUNAI_PROJECT}."
  exit 0
fi

COUNT=$(echo "${JOBS}" | wc -l | tr -d ' ')
echo "Project: ${RUNAI_PROJECT}"
echo "Filter : ${LABEL}"
echo "Found  : ${COUNT} job(s) to delete"
echo "${JOBS}" | sed 's/^/  - /'
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[--dry-run] no deletions performed."
  exit 0
fi

# Confirm for non-default destructive filters.
if [[ "${FILTER}" == "all" || "${FILTER,,}" != "succeeded" && "${FILTER,,}" != "failed" ]]; then
  read -r -p "Confirm delete ${COUNT} job(s) from ${RUNAI_PROJECT}? [y/N] " ans
  case "${ans}" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

for j in ${JOBS}; do
  runai delete job "${j}" -p "${RUNAI_PROJECT}" || echo "[warn] failed to delete ${j}" >&2
done

echo ""
echo "Done. Remaining jobs:"
runai list -p "${RUNAI_PROJECT}" | tail -n +3 | wc -l | xargs -I{} echo "  {} job(s) still in the project"
