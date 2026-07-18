#!/usr/bin/env python3
"""Resumable, validated OMX4D baker for a trusted OmniX batch plan."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from omx4d_tools.config import GIB, MIB, IngestionLimits
from omx4d_tools.converter import ConversionOptions, convert_pt_file
from omx4d_tools.omx4d import read_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINT_BUDGET = 500_000
DEFAULT_DYNAMIC_FRACTION = 0.8
DEFAULT_DYNAMIC_THRESHOLD = 0.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_to_resolution(image: Image.Image, width: int, height: int) -> np.ndarray:
    input_width, input_height = image.size
    scale = max(width / input_width, height / input_height) + 1e-8
    resize_width = int(np.floor(input_width * scale))
    resize_height = int(np.floor(input_height * scale))
    resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
    resample = resampling.LANCZOS if scale < 1 else resampling.BICUBIC
    image = image.resize((resize_width, resize_height), resample=resample)
    left = (resize_width - width) / 2
    top = (resize_height - height) / 2
    image = image.crop((left, top, left + width, top + height))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_source_rgb(
    image_root: Path,
    source_views: int,
    width: int,
    height: int,
) -> torch.Tensor:
    candidates = [image_root / "video_00", image_root]
    frame_paths: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            frame_paths = sorted(
                path
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if frame_paths:
                break
    if len(frame_paths) != source_views:
        raise RuntimeError(
            f"{image_root} has {len(frame_paths)} RGB frames; expected {source_views}"
        )
    frames: list[np.ndarray] = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(crop_to_resolution(image, width, height))
    rgb = np.stack(frames)
    if rgb.shape != (source_views, height, width, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"Unexpected RGB tensor {rgb.shape} {rgb.dtype}")
    return torch.from_numpy(rgb).contiguous()


def sha256(path: Path, block_size: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def expected_file_size(manifest: dict[str, Any]) -> int:
    return max(
        int(descriptor["offset"]) + int(descriptor["byteLength"])
        for descriptor in manifest["attributes"].values()
    )


def valid_existing(
    path: Path,
    fps: float,
    frame_count: int,
    point_budget: int,
    dynamic_fraction: float,
    dynamic_count: int,
    spatial_count: int,
    candidate_source_view_count: int,
    source_pt_sha256: str,
    sidecar_path: Path | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = read_manifest(path)
        sampling = manifest.get("sampling", {})
        valid = (
            manifest.get("schemaVersion") == 1
            and manifest.get("fps") == fps
            and manifest.get("frameCount") == frame_count
            and manifest.get("sourceViewCount") == frame_count
            and manifest.get("pointCount") == point_budget
            and sampling.get("dynamicThreshold") == DEFAULT_DYNAMIC_THRESHOLD
            and sampling.get("dynamicReservedFraction") == dynamic_fraction
            and sampling.get("dynamicSelectedPointCount") == dynamic_count
            and sampling.get("dynamicRanking")
            == "global-descending-stable-identity-tiebreak"
            and isinstance(sampling.get("dynamicScoreCutoff"), (int, float))
            and sampling.get("spatialSelectedPointCount") == spatial_count
            and sampling.get("spatialDistribution")
            == "normalized-frame-zero-3d-voxel"
            and sampling.get("candidateSourceViewCount")
            == candidate_source_view_count
            and path.stat().st_size == expected_file_size(manifest)
            and not any(
                "Source RGB was not supplied" in warning
                for warning in manifest.get("warnings", [])
            )
        )
        if not valid:
            return False
        if sidecar_path is None:
            return True
        if not sidecar_path.is_file():
            return False
        sidecar = json.loads(sidecar_path.read_text())
        return (
            sidecar.get("sha256") == sha256(path)
            and sidecar.get("source_pt_sha256") == source_pt_sha256
        )
    except Exception:
        return False


def limits(point_budget: int) -> IngestionLimits:
    return IngestionLimits(
        max_upload_bytes=5 * GIB,
        max_archive_uncompressed_bytes=5 * GIB,
        max_total_tensor_bytes=5 * GIB,
        max_output_bytes=512 * MIB,
        max_source_views=64,
        max_frames=64,
        max_source_pixels=16_000_000,
        max_point_budget=point_budget,
        max_zip_entries=512,
        max_compression_ratio=200.0,
        finite_check_chunk_elements=8_000_000,
    )


def bake_one(
    repo_root: Path,
    output_root: Path,
    video_id: str,
    chunk: dict[str, Any],
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    point_budget: int,
    dynamic_fraction: float,
    ordinal: int,
    total: int,
    force: bool,
) -> dict[str, Any]:
    chunk_id = str(chunk["id"])
    source_path = repo_root / str(chunk["prediction_file"])
    image_root = repo_root / str(chunk["input_dir"])
    output_path = output_root / video_id / f"{chunk_id}.omx4d"
    sidecar_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    status_path = source_path.parent / "status.json"
    source_status = json.loads(status_path.read_text())
    expected_source = str(source_path.relative_to(repo_root))
    if (
        source_status.get("state") != "complete"
        or source_status.get("prediction_file") != expected_source
        or int(source_status.get("pt_bytes", -1)) != source_path.stat().st_size
        or not isinstance(source_status.get("pt_sha256"), str)
    ):
        raise RuntimeError(f"Source status validation failed for {source_path}")

    dynamic_count = int(round(point_budget * dynamic_fraction))
    spatial_count = point_budget - dynamic_count
    candidate_source_view_count = int(chunk["valid_frames"])
    if not force and valid_existing(
        output_path,
        fps,
        frame_count,
        point_budget,
        dynamic_fraction,
        dynamic_count,
        spatial_count,
        candidate_source_view_count,
        source_status["pt_sha256"],
        sidecar_path,
    ):
        print(
            json.dumps(
                {
                    "event": "skip",
                    "ordinal": ordinal,
                    "total": total,
                    "video": video_id,
                    "chunk": chunk_id,
                    "bytes": output_path.stat().st_size,
                }
            ),
            flush=True,
        )
        record = json.loads(sidecar_path.read_text())
        record["source_pt_sha256"] = source_status["pt_sha256"]
        record["inference_fingerprint"] = source_status.get("inference_fingerprint")
        atomic_json(sidecar_path, record)
        return record

    partial_path = output_path.with_suffix(".omx4d.partial")
    partial_path.unlink(missing_ok=True)
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "start",
                "ordinal": ordinal,
                "total": total,
                "video": video_id,
                "chunk": chunk_id,
                "source": str(source_path.relative_to(repo_root)),
            }
        ),
        flush=True,
    )

    source_rgb = load_source_rgb(image_root, frame_count, width, height)
    name = (
        f"{video_id} {chunk_id} "
        f"{float(chunk['start_time_s']):.3f}-{float(chunk['end_time_s']):.3f}s"
    )
    try:
        result = convert_pt_file(
            source_path,
            partial_path,
            options=ConversionOptions(
                point_budget=point_budget,
                fps=fps,
                name=name,
                dynamic_threshold=DEFAULT_DYNAMIC_THRESHOLD,
                dynamic_reserved_fraction=dynamic_fraction,
                candidate_source_view_count=candidate_source_view_count,
            ),
            limits=limits(point_budget),
            source_rgb=source_rgb,
        )
        os.replace(partial_path, output_path)
    finally:
        partial_path.unlink(missing_ok=True)
        del source_rgb
        gc.collect()

    manifest = read_manifest(output_path)
    sampling = manifest.get("sampling", {})
    if (
        manifest.get("pointCount") != point_budget
        or sampling.get("dynamicSelectedPointCount") != dynamic_count
        or sampling.get("spatialSelectedPointCount") != spatial_count
    ):
        raise RuntimeError(
            f"Sampling split is not {dynamic_count} dynamic / {spatial_count} spatial"
        )
    if not valid_existing(
        output_path,
        fps,
        frame_count,
        point_budget,
        dynamic_fraction,
        dynamic_count,
        spatial_count,
        candidate_source_view_count,
        source_status["pt_sha256"],
    ):
        raise RuntimeError(f"Post-write validation failed for {output_path}")
    elapsed = time.monotonic() - started
    record = {
        "schema": "omnix.omx4d-bake.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": video_id,
        "chunk": chunk_id,
        "start_time_s": float(chunk["start_time_s"]),
        "end_time_s": float(chunk["end_time_s"]),
        "valid_frames": int(chunk["valid_frames"]),
        "pad_frames": int(chunk["pad_frames"]),
        "source_pt": str(source_path.relative_to(repo_root)),
        "source_pt_sha256": source_status["pt_sha256"],
        "inference_fingerprint": source_status.get("inference_fingerprint"),
        "source_rgb": str(image_root.relative_to(repo_root)),
        "output": str(output_path.relative_to(repo_root)),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        "elapsed_seconds": elapsed,
        "point_count": int(manifest["pointCount"]),
        "frame_count": int(manifest["frameCount"]),
        "fps": float(manifest["fps"]),
        "source_resolution": [width, height],
        "bounds": manifest["bounds"],
        "sampling": manifest["sampling"],
        "warnings": manifest.get("warnings", []),
    }
    atomic_json(sidecar_path, record)
    print(
        json.dumps(
            {
                "event": "complete",
                "ordinal": ordinal,
                "total": total,
                "video": video_id,
                "chunk": chunk_id,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "elapsed_seconds": round(elapsed, 3),
            }
        ),
        flush=True,
    )
    del result
    gc.collect()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video")
    parser.add_argument("--chunk")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--point-budget", type=int, default=DEFAULT_POINT_BUDGET)
    parser.add_argument(
        "--dynamic-reserved-fraction",
        type=float,
        default=DEFAULT_DYNAMIC_FRACTION,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "omnix.batch-plan.v1":
        raise ValueError(f"Unsupported plan schema: {plan.get('schema')!r}")
    if args.point_budget <= 0:
        raise ValueError("--point-budget must be positive")
    if not 0.0 <= args.dynamic_reserved_fraction <= 1.0:
        raise ValueError("--dynamic-reserved-fraction must be in [0, 1]")
    fps = float(plan["sampling"]["fps"])
    frame_count = int(plan["sampling"]["frames_per_chunk"])
    width = int(plan["sampling"]["model_width"])
    height = int(plan["sampling"]["model_height"])
    if args.output_root is None:
        batch_root = repo_root / str(plan["output_root"])
        dynamic_percent = round(args.dynamic_reserved_fraction * 100)
        output_root = (
            batch_root
            / f"omx4d_{args.point_budget // 1000}k_{dynamic_percent}d"
            f"{100 - dynamic_percent}s"
        )
    else:
        output_root = (
            args.output_root
            if args.output_root.is_absolute()
            else repo_root / args.output_root
        )
    videos = [video for video in plan["videos"] if not args.video or video["id"] == args.video]
    if not videos:
        raise SystemExit(f"Video not found: {args.video}")
    work = [
        (str(video["id"]), chunk)
        for video in videos
        for chunk in video["chunks"]
        if not args.chunk or chunk["id"] == args.chunk
    ]
    if not work:
        raise SystemExit(f"No chunks match --video={args.video!r} --chunk={args.chunk!r}")
    if args.limit is not None:
        work = work[: args.limit]
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)

    records: list[dict[str, Any]] = []
    for ordinal, (video_id, chunk) in enumerate(work, start=1):
        records.append(
            bake_one(
                repo_root,
                output_root,
                video_id,
                chunk,
                fps,
                frame_count,
                width,
                height,
                args.point_budget,
                args.dynamic_reserved_fraction,
                ordinal,
                len(work),
                args.force,
            )
        )
    summary = {
        "schema": "omnix.omx4d-worker-summary.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "video": args.video,
        "point_budget": args.point_budget,
        "dynamic_reserved_fraction": args.dynamic_reserved_fraction,
        "model_resolution": [width, height],
        "fps": fps,
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "records": records,
    }
    suffix = "_".join(value for value in (args.video, args.chunk) if value) or "all"
    atomic_json(output_root / f"worker_summary_{suffix}.json", summary)
    print(json.dumps({"event": "worker_complete", **{k: summary[k] for k in ("video", "files", "bytes")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
