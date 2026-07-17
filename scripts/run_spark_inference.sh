#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${OMNIX_IMAGE:-omnix-dgx-spark:latest}"
CHECKPOINT="${REPO_ROOT}/pretrained_weight/eccv_release.ckpt"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  echo "Download yanqinJiang/omnix/eccv_release.ckpt before running inference." >&2
  exit 1
fi

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "${REPO_ROOT}:/workspace/OmniX" \
  --workdir /workspace/OmniX \
  "${IMAGE_NAME}" \
  python visualize_simple.py \
    +experiment=release_train \
    +paths.image_folder=images/test_deer \
    +paths.checkpoint_path=pretrained_weight/eccv_release.ckpt \
    +paths.output_path=outputs/test_deer_output \
    +paths.render_scale=1 \
    +paths.orbit_frames=8 \
    "$@"
