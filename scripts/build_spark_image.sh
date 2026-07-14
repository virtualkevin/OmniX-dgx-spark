#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${OMNIX_IMAGE:-omnix-dgx-spark:latest}"

docker build \
  --file "${REPO_ROOT}/Dockerfile.spark" \
  --tag "${IMAGE_NAME}" \
  "${REPO_ROOT}"
