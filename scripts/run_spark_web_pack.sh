#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch-plan.json> [packer arguments ...]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$1"
shift

if [[ "${MANIFEST}" = /* || "${MANIFEST}" == *".."* ]]; then
  echo "Batch manifest must be a repository-relative path without '..'" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/${MANIFEST}" ]]; then
  echo "Missing batch manifest: ${REPO_ROOT}/${MANIFEST}" >&2
  exit 1
fi
MANIFEST_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["docker_image"])' "${REPO_ROOT}/${MANIFEST}")"
EXPECTED_IMAGE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["docker_image_id"])' "${REPO_ROOT}/${MANIFEST}")"
IMAGE_NAME="${OMNIX_IMAGE:-${MANIFEST_IMAGE}}"
ACTUAL_IMAGE_ID="$(docker image inspect --format='{{.Id}}' "${IMAGE_NAME}")"
if [[ "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Docker image ID differs from the batch manifest" >&2
  exit 1
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "${REPO_ROOT}:/workspace/OmniX" \
  --workdir /workspace/OmniX \
  "${IMAGE_NAME}" \
  python scripts/pack_omnix_web.py \
    --manifest "${MANIFEST}" \
    "$@"
