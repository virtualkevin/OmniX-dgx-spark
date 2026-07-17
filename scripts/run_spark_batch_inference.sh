#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch-plan.json> [+paths.max_chunks=N] [+paths.only_video=ID] [+paths.only_chunk=ID]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$1"
shift

for override in "$@"; do
  case "${override}" in
    +paths.max_chunks=*|+paths.only_video=*|+paths.only_chunk=*) ;;
    *)
      echo "Unsupported override for a resumable batch: ${override}" >&2
      echo "Only max_chunks, only_video, and only_chunk filters are allowed." >&2
      exit 2
      ;;
  esac
done

if [[ "${MANIFEST}" = /* || "${MANIFEST}" == *".."* ]]; then
  echo "Batch manifest must be a repository-relative path without '..'" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/${MANIFEST}" ]]; then
  echo "Missing batch manifest: ${REPO_ROOT}/${MANIFEST}" >&2
  exit 1
fi
CHECKPOINT_REL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["checkpoint"])' "${REPO_ROOT}/${MANIFEST}")"
MANIFEST_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["docker_image"])' "${REPO_ROOT}/${MANIFEST}")"
CHECKPOINT="${REPO_ROOT}/${CHECKPOINT_REL}"
IMAGE_NAME="${OMNIX_IMAGE:-${MANIFEST_IMAGE}}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
IMAGE_ID="$(docker image inspect --format='{{.Id}}' "${IMAGE_NAME}")"

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env OMNIX_IMAGE_ID="${IMAGE_ID}" \
  --volume "${REPO_ROOT}:/workspace/OmniX" \
  --workdir /workspace/OmniX \
  "${IMAGE_NAME}" \
  python batch_inference.py \
    +experiment=release_train \
    +paths.batch_manifest="${MANIFEST}" \
    +paths.checkpoint_path="${CHECKPOINT_REL}" \
    "$@"
