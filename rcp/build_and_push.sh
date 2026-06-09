#!/usr/bin/env bash
# Build the antisemitism-detection Docker image and push to EPFL's RCP registry.
#
# Prereqs:
#   - Logged in to the registry: `docker login registry.rcp.epfl.ch`
#   - Run from the repo root: `bash rcp/build_and_push.sh`
#
# Override the tag with TAG=<tag> bash rcp/build_and_push.sh (default: latest).

set -euo pipefail

TAG="${TAG:-latest}"
IMAGE="registry.rcp.epfl.ch/ee-559-case/antisemitism:${TAG}"

echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" -f rcp/Dockerfile rcp/

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo "Done. Use IMAGE=${IMAGE} with rcp/submit.sh (or export IMAGE to override default)."
