#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <batch-plan.json> <omx4d-root>" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="$1"
OMX4D_ROOT="$2"

for value in "${PLAN}" "${OMX4D_ROOT}"; do
  if [[ "${value}" = /* || "${value}" == *".."* ]]; then
    echo "Paths must be repository-relative and must not contain '..'" >&2
    exit 2
  fi
done
if [[ ! -f "${REPO_ROOT}/${PLAN}" ]]; then
  echo "Missing batch plan: ${REPO_ROOT}/${PLAN}" >&2
  exit 1
fi

MANIFEST_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["docker_image"])' "${REPO_ROOT}/${PLAN}")"
EXPECTED_IMAGE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["docker_image_id"])' "${REPO_ROOT}/${PLAN}")"
IMAGE_NAME="${OMNIX_IMAGE:-${MANIFEST_IMAGE}}"
ACTUAL_IMAGE_ID="$(docker image inspect --format='{{.Id}}' "${IMAGE_NAME}")"
if [[ "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Docker image ID differs from the batch plan" >&2
  exit 1
fi

docker run --rm \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "${REPO_ROOT}:/workspace/OmniX" \
  --workdir /workspace/OmniX \
  "${IMAGE_NAME}" \
  python scripts/verify_omx4d_batch.py \
    --plan "${PLAN}" \
    --omx4d-root "${OMX4D_ROOT}"
