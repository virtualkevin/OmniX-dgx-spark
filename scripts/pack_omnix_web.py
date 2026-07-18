#!/usr/bin/env python3
"""Convert raw OmniX PT shards into browser-streamable typed-array shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMELINE_SCHEMA = "omnix.timeline.v1"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
RAW_DTYPES = {
    "trajectory": torch.float32,
    "camera_pose": torch.float32,
    "intrinsics": torch.float32,
    "pts3d_dynamic_score": torch.float32,
}


def repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must stay inside the repository: {value}") from error
    return path


def contained_path(root: Path, candidate: Path, label: str) -> Path:
    root = root.resolve()
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside {root}: {candidate}") from error
    return path


def validate_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase slug: {value!r}")
    return value


def shard_profile(point_count: int) -> str:
    schema_slug = TIMELINE_SCHEMA.replace(".", "-")
    return f"{schema_slug}-points-{point_count:06d}"


def expected_raw_shapes(
    frame_count: int, height: int, width: int
) -> dict[str, list[int]]:
    return {
        "trajectory": [frame_count, frame_count, height, width, 3],
        "camera_pose": [frame_count, 3, 4],
        "intrinsics": [frame_count, 3, 3],
        "pts3d_dynamic_score": [frame_count, height, width],
    }


def tensor_is_finite(
    tensor: torch.Tensor, block_elements: int = 16 * 1024 * 1024
) -> bool:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), block_elements):
        if not torch.isfinite(flat[start : start + block_elements]).all().item():
            return False
    return True


def validate_raw_predictions(
    raw: Any, frame_count: int, height: int, width: int, pt_path: Path
) -> None:
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected a tensor dictionary in {pt_path}")
    for key, shape in expected_raw_shapes(frame_count, height, width).items():
        value = raw.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Missing tensor {key!r} in {pt_path}")
        if list(value.shape) != shape:
            raise RuntimeError(
                f"Unexpected {key} shape in {pt_path}: {list(value.shape)}, expected {shape}"
            )
        if value.dtype != RAW_DTYPES[key]:
            raise RuntimeError(
                f"Unexpected {key} dtype in {pt_path}: {value.dtype}, expected {RAW_DTYPES[key]}"
            )
        if not tensor_is_finite(value):
            raise RuntimeError(f"Non-finite values in {key} from {pt_path}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.tofile(temporary)
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def crop_to_resolution(image: PIL.Image.Image, width: int, height: int) -> np.ndarray:
    input_width, input_height = image.size
    scale = max(width / input_width, height / input_height) + 1e-8
    resize_width = int(np.floor(input_width * scale))
    resize_height = int(np.floor(input_height * scale))
    if hasattr(PIL.Image, "Resampling"):
        resample = (
            PIL.Image.Resampling.LANCZOS if scale < 1 else PIL.Image.Resampling.BICUBIC
        )
    else:
        resample = PIL.Image.LANCZOS if scale < 1 else PIL.Image.BICUBIC
    image = image.resize((resize_width, resize_height), resample=resample)
    left = (resize_width - width) / 2
    top = (resize_height - height) / 2
    image = image.crop((left, top, left + width, top + height))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def select_points(dynamic_score: torch.Tensor, point_count: int) -> torch.Tensor:
    flat_score = dynamic_score.reshape(-1).float()
    total = flat_score.numel()
    point_count = min(point_count, total)
    uniform_count = min(point_count // 2, total)
    uniform = torch.linspace(0, total - 1, steps=uniform_count).round().long().unique()

    selected = torch.zeros(total, dtype=torch.bool)
    selected[uniform] = True
    priority = torch.argsort(flat_score, descending=True)
    dynamic = priority[~selected[priority]][: point_count - uniform.numel()]
    indices = torch.cat([uniform, dynamic])
    if indices.numel() < point_count:
        selected[dynamic] = True
        remaining = torch.arange(total)[~selected]
        indices = torch.cat([indices, remaining[: point_count - indices.numel()]])
    return indices


def binary_descriptor(
    web_root: Path,
    path: Path,
    dtype: str,
    shape: list[int],
    axes: list[str],
) -> dict[str, Any]:
    return {
        "path": path.relative_to(web_root).as_posix(),
        "dtype": dtype,
        "shape": shape,
        "axes": axes,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def inference_fingerprint(
    manifest: dict[str, Any],
    video: dict[str, Any],
    chunk: dict[str, Any],
) -> str:
    payload = {
        "schema": manifest["schema"],
        "sampling": manifest["sampling"],
        "provenance": manifest["provenance"],
        "video": {
            "id": video["id"],
            "source_sha256": video["source_sha256"],
            "crop_filter": video.get("crop_filter"),
        },
        "chunk": {
            "id": chunk["id"],
            "source_sample_indices": chunk["source_sample_indices"],
            "valid_frames": chunk["valid_frames"],
            "pad_frames": chunk["pad_frames"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def completed_chunk_error(
    manifest: dict[str, Any],
    video: dict[str, Any],
    chunk: dict[str, Any],
    frame_count: int,
    height: int,
    width: int,
) -> str | None:
    pt_path = repo_path(chunk["prediction_file"])
    output_dir = repo_path(chunk["output_dir"])
    status_path = output_dir / "status.json"
    summary_path = output_dir / "prediction_summary.json"
    if not pt_path.is_file():
        return "PT missing"
    if not status_path.is_file() or not summary_path.is_file():
        return "status or summary missing"
    status = json.loads(status_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (
        status.get("state") != "complete"
        or status.get("pt_bytes") != pt_path.stat().st_size
    ):
        return "status is not complete or byte size differs"
    if status.get("inference_fingerprint") != inference_fingerprint(
        manifest, video, chunk
    ):
        return "inference fingerprint differs from batch manifest"
    expected = {
        "trajectory": [1, frame_count, frame_count, height, width, 3],
        "camera_pose": [1, frame_count, 3, 4],
        "intrinsics": [1, frame_count, 3, 3],
        "pts3d_dynamic_score": [1, frame_count, height, width],
    }
    for key, shape in expected.items():
        if summary.get(key, {}).get("shape") != shape:
            return f"unexpected {key} shape"
        if summary.get(key, {}).get("finite_fraction") != 1.0:
            return f"non-finite {key} values"
    if not status.get("pt_sha256"):
        return "PT checksum missing from status"
    if status["pt_sha256"] != sha256(pt_path):
        return "PT checksum differs from status"
    return None


def pack_chunk(
    web_root: Path,
    profile_id: str,
    video_id: str,
    chunk: dict[str, Any],
    point_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    pt_path = repo_path(chunk["prediction_file"])
    raw = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    validate_raw_predictions(raw, frame_count, height, width, pt_path)
    trajectory = raw["trajectory"]
    _, target_count, _, _, _ = trajectory.shape

    valid_frames = int(chunk["valid_frames"])
    reference_index = min(valid_frames // 2, frame_count - 1)
    indices = select_points(raw["pts3d_dynamic_score"][reference_index], point_count)
    chunk_dir = contained_path(
        web_root,
        web_root / "chunks" / profile_id / video_id / chunk["id"],
        "Web shard output",
    )

    positions_path = chunk_dir / "positions.f32.bin"
    pixels_path = chunk_dir / "pixels.u32.bin"
    colors_path = chunk_dir / "colors.u8.bin"
    dynamic_path = chunk_dir / "dynamic.u8.bin"
    camera_path = chunk_dir / "camera_pose.f32.bin"
    intrinsics_path = chunk_dir / "intrinsics.f32.bin"

    reference_trajectory = trajectory[reference_index].reshape(
        target_count, height * width, 3
    )
    positions = (
        reference_trajectory.index_select(1, indices).numpy().astype("<f4", copy=False)
    )
    pixels = indices.numpy().astype("<u4", copy=False)
    dynamic = (
        raw["pts3d_dynamic_score"][reference_index]
        .reshape(-1)
        .index_select(0, indices)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    input_root = repo_path(chunk["input_dir"])
    frame_paths = sorted((input_root / "video_00").glob("*.jpg"))
    if len(frame_paths) != frame_count:
        raise RuntimeError(
            f"Expected {frame_count} input frames in {input_root}, found {len(frame_paths)}"
        )
    with PIL.Image.open(frame_paths[reference_index]) as reference_image:
        colors_image = crop_to_resolution(reference_image, width, height)
    colors = colors_image.reshape(-1, 3)[pixels]
    camera_pose = raw["camera_pose"].numpy().astype("<f4", copy=False)
    intrinsics = raw["intrinsics"].numpy().astype("<f4", copy=False)

    atomic_array(positions_path, positions)
    atomic_array(pixels_path, pixels)
    atomic_array(colors_path, colors)
    atomic_array(dynamic_path, dynamic)
    atomic_array(camera_path, camera_pose)
    atomic_array(intrinsics_path, intrinsics)

    status_path = repo_path(chunk["output_dir"]) / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    del raw, trajectory, reference_trajectory, positions
    gc.collect()

    return {
        "id": chunk["id"],
        "index": chunk["index"],
        "start_time_s": chunk["start_time_s"],
        "end_time_s": chunk["end_time_s"],
        "sample_times_s": chunk["sample_times_s"],
        "valid_frames": valid_frames,
        "pad_frames": chunk["pad_frames"],
        "raw_pt": {
            "repo_path": chunk["prediction_file"],
            "bytes": pt_path.stat().st_size,
            "sha256": status.get("pt_sha256"),
            "role": "provenance-only-not-required-by-browser",
        },
        "web": {
            "reference_index": reference_index,
            "point_count": int(indices.numel()),
            "positions": binary_descriptor(
                web_root,
                positions_path,
                "float32-le",
                [target_count, int(indices.numel()), 3],
                ["time", "point", "xyz"],
            ),
            "pixel_indices": binary_descriptor(
                web_root, pixels_path, "uint32-le", [int(indices.numel())], ["point"]
            ),
            "colors": binary_descriptor(
                web_root,
                colors_path,
                "uint8",
                [int(indices.numel()), 3],
                ["point", "rgb"],
            ),
            "dynamic_score": {
                **binary_descriptor(
                    web_root, dynamic_path, "uint8", [int(indices.numel())], ["point"]
                ),
                "scale": 1 / 255,
            },
            "camera_pose": binary_descriptor(
                web_root,
                camera_path,
                "float32-le",
                list(camera_pose.shape),
                ["time", "row", "column"],
            ),
            "intrinsics": binary_descriptor(
                web_root,
                intrinsics_path,
                "float32-le",
                list(intrinsics.shape),
                ["time", "row", "column"],
            ),
        },
        "alignment": {"status": "chunk-local", "segment": chunk["index"]},
    }


def validate_batch_manifest(batch: dict[str, Any]) -> int:
    videos = batch.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Batch manifest must contain at least one video")
    seen_videos: set[str] = set()
    total_chunks = 0
    for video in videos:
        video_id = validate_id(video.get("id"), "Video ID")
        if video_id in seen_videos:
            raise ValueError(f"Duplicate video ID: {video_id}")
        seen_videos.add(video_id)
        chunks = video.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"Video {video_id} must contain at least one chunk")
        seen_chunks: set[str] = set()
        seen_indices: set[int] = set()
        for chunk in chunks:
            chunk_id = validate_id(chunk.get("id"), f"Chunk ID in {video_id}")
            if chunk_id in seen_chunks:
                raise ValueError(f"Duplicate chunk ID in {video_id}: {chunk_id}")
            seen_chunks.add(chunk_id)
            index = chunk.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in seen_indices
            ):
                raise ValueError(
                    f"Invalid or duplicate chunk index in {video_id}: {index!r}"
                )
            seen_indices.add(index)
        total_chunks += len(chunks)
    declared_total = batch.get("total_chunks")
    if declared_total != total_chunks:
        raise ValueError(
            f"Batch total_chunks is {declared_total!r}, expected {total_chunks}"
        )
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", required=True, help="Batch plan inside the repository"
    )
    parser.add_argument("--output-root", help="Defaults to <batch output root>/web")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--video")
    parser.add_argument("--limit", type=int)
    readiness = parser.add_mutually_exclusive_group()
    readiness.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail if any selected PT is not valid (default)",
    )
    readiness.add_argument(
        "--allow-incomplete",
        dest="strict",
        action="store_false",
        help="Skip invalid PTs and write manifest.partial.json",
    )
    parser.set_defaults(strict=True)
    args = parser.parse_args()
    if args.points <= 0:
        raise ValueError("--points must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.video is not None:
        validate_id(args.video, "--video")

    manifest_path = repo_path(args.manifest)
    batch = json.loads(manifest_path.read_text())
    if batch.get("schema") != "omnix.batch-plan.v1":
        raise ValueError(f"Unsupported batch manifest schema: {batch.get('schema')!r}")
    source_chunk_count = validate_batch_manifest(batch)
    batch_root = repo_path(batch["output_root"])
    sampling = batch.get("sampling", {})
    frame_count = sampling.get("frames_per_chunk")
    model_width = sampling.get("model_width", 504)
    model_height = sampling.get("model_height", 280)
    fps = sampling.get("fps")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count <= 0
    ):
        raise ValueError("Batch frame count must be a positive integer")
    if (
        not isinstance(model_width, int)
        or isinstance(model_width, bool)
        or model_width <= 0
        or model_width % 14 != 0
        or not isinstance(model_height, int)
        or isinstance(model_height, bool)
        or model_height <= 0
        or model_height % 14 != 0
    ):
        raise ValueError(
            "Batch model dimensions must be positive integers divisible by 14"
        )
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Batch sampling FPS must be finite and positive")
    if args.points > model_width * model_height:
        raise ValueError(
            f"--points cannot exceed the {model_width * model_height} model pixels"
        )
    web_candidate = (
        repo_path(args.output_root) if args.output_root else batch_root / "web"
    )
    web_root = contained_path(REPO_ROOT, web_candidate, "Web output root")
    profile_id = shard_profile(args.points)
    selected = [
        (video["id"], video, chunk)
        for video in batch["videos"]
        if args.video is None or video["id"] == args.video
        for chunk in video["chunks"]
    ]
    if args.video is not None and not selected:
        raise ValueError(f"No batch video matches --video={args.video!r}")
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No chunks selected for web packing")

    packed_by_video: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    for ordinal, (video_id, _video, chunk) in enumerate(selected, start=1):
        readiness_error = completed_chunk_error(
            batch, _video, chunk, frame_count, model_height, model_width
        )
        if readiness_error is not None:
            if args.strict:
                raise RuntimeError(
                    f"Cannot pack {video_id}/{chunk['id']}: {readiness_error}"
                )
            print(
                f"[{ordinal}/{len(selected)}] SKIP {video_id}/{chunk['id']}: "
                f"{readiness_error}",
                flush=True,
            )
            skipped.append(
                {
                    "video_id": video_id,
                    "chunk_id": chunk["id"],
                    "reason": readiness_error,
                }
            )
            continue
        print(f"[{ordinal}/{len(selected)}] PACK {video_id}/{chunk['id']}", flush=True)
        packed_by_video.setdefault(video_id, []).append(
            pack_chunk(
                web_root,
                profile_id,
                video_id,
                chunk,
                args.points,
                frame_count,
                model_height,
                model_width,
            )
        )

    videos = []
    for source_video in batch["videos"]:
        chunks = packed_by_video.get(source_video["id"], [])
        if chunks:
            source_ids = [chunk["id"] for chunk in source_video["chunks"]]
            packed_ids = [chunk["id"] for chunk in chunks]
            videos.append(
                {
                    "id": source_video["id"],
                    "source_repo_path": source_video["source"],
                    "source_sha256": source_video["source_sha256"],
                    "source_crop_filter": source_video.get("crop_filter"),
                    "duration_s": float(source_video["source_probe"]["duration"]),
                    "source_chunk_count": len(source_ids),
                    "packed_chunk_count": len(packed_ids),
                    "timeline_complete": packed_ids == source_ids,
                    "chunks": chunks,
                }
            )

    packed_chunk_count = sum(len(video["chunks"]) for video in videos)
    if packed_chunk_count == 0:
        raise RuntimeError("No valid chunks were available to pack")
    filters_used = args.video is not None or args.limit is not None
    package_complete = (
        not filters_used and not skipped and packed_chunk_count == source_chunk_count
    )
    web_manifest = {
        "schema": TIMELINE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_batch_manifest_repo_path": manifest_path.relative_to(
            REPO_ROOT
        ).as_posix(),
        "sampling": batch["sampling"],
        "packaging": {
            "complete": package_complete,
            "profile_id": profile_id,
            "source_chunk_count": source_chunk_count,
            "selected_chunk_count": len(selected),
            "packed_chunk_count": packed_chunk_count,
            "skipped_chunk_count": len(skipped),
            "filters": {"video": args.video, "limit": args.limit},
            "skipped": skipped,
        },
        "point_selection": {
            "maximum_points": args.points,
            "strategy": "half-uniform-flattened-raster-plus-highest-dynamic-score",
        },
        "binary_contract": {
            "endianness": "little",
            "array_order": "C-row-major",
            "pixel_indexing": "pixel=y*model_width+x",
            "reference_attributes": "colors and dynamic scores use web.reference_index",
            "paths": "relative to this manifest",
            "shard_profile": profile_id,
        },
        "coordinate_contract": {
            "positions": "predicted chunk-local world coordinates",
            "camera_pose": "camera-to-world in the same chunk-local gauge",
            "units": "model-relative",
            "camera_basis": "OpenCV: +x right, +y down, +z forward",
            "camera_matrix_layout": "row-major 3x4 camera-to-world",
            "threejs_conversion": {
                "basis_diagonal": [1, -1, -1, 1],
                "points": "p_three=C*p_opencv",
                "camera": "c2w_three=C*c2w_opencv*C",
            },
            "cross_chunk_alignment": "none",
            "boundary_behavior": "reset-or-crossfade",
        },
        "preprocessing": {
            "model_resolution": [model_width, model_height],
            "image_policy": "aspect-preserving resize then center crop",
            "source_crop_filter_per_video": True,
        },
        "videos": videos,
        "packed_chunk_count": packed_chunk_count,
    }
    is_partial = not package_complete
    manifest_output = web_root / (
        "manifest.partial.json" if is_partial else "manifest.json"
    )
    atomic_json(manifest_output, web_manifest)
    print(
        f"Wrote {manifest_output} with {web_manifest['packed_chunk_count']} chunks",
        flush=True,
    )


if __name__ == "__main__":
    main()
