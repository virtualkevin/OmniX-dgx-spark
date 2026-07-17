"""Resumable multi-chunk OmniX inference with one model/checkpoint load."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from visualize_simple import (  # noqa: E402
    load_images_from_folder,
    load_inference_weights,
    summarize_predictions,
    validate_geometry,
)


REQUIRED_KEYS = ["trajectory", "camera_pose", "intrinsics", "pts3d_dynamic_score"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes repository: {value}") from error
    return path


def flatten_chunks(
    manifest: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(video, chunk) for video in manifest["videos"] for chunk in video["chunks"]]


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


def expected_raw_shapes(frame_count: int) -> dict[str, list[int]]:
    return {
        "trajectory": [frame_count, frame_count, 280, 504, 3],
        "camera_pose": [frame_count, 3, 4],
        "intrinsics": [frame_count, 3, 3],
        "pts3d_dynamic_score": [frame_count, 280, 504],
    }


def files_are_identical(
    first: Path, second: Path, digest_cache: dict[Path, str]
) -> bool:
    try:
        if os.path.samefile(first, second):
            return True
    except OSError:
        pass
    if first.stat().st_size != second.stat().st_size:
        return False

    def cached_digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = sha256(path)
        return digest_cache[path]

    return cached_digest(first) == cached_digest(second)


def validate_chunk_inputs(
    repo_root: Path,
    video: dict[str, Any],
    chunk: dict[str, Any],
    frame_count: int,
    digest_cache: dict[Path, str],
) -> None:
    label = f"{video['id']}/{chunk['id']}"
    input_root = resolve_repo_path(repo_root, chunk["input_dir"])
    sampled_root = resolve_repo_path(repo_root, video["sampled_frames_dir"])
    if not input_root.is_dir() or not sampled_root.is_dir():
        raise RuntimeError(f"{label} input or sampled-frame directory is missing")

    video_directories = sorted(
        path.name for path in input_root.iterdir() if path.is_dir()
    )
    if video_directories != ["video_00"]:
        raise RuntimeError(
            f"{label} must contain exactly the video_00 directory, found "
            f"{video_directories}"
        )
    frame_root = input_root / "video_00"
    actual_names = sorted(
        path.name
        for path in frame_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    expected_names = [f"frame_{index:04d}.jpg" for index in range(frame_count)]
    if actual_names != expected_names:
        raise RuntimeError(
            f"{label} frame filenames/count differ from the fixed {frame_count}-frame layout"
        )

    source_indices = chunk.get("source_sample_indices")
    if (
        not isinstance(source_indices, list)
        or len(source_indices) != frame_count
        or any(type(index) is not int or index < 0 for index in source_indices)
    ):
        raise RuntimeError(f"{label} has invalid source_sample_indices")
    for local_index, source_index in enumerate(source_indices):
        chunk_frame = frame_root / expected_names[local_index]
        sampled_frame = sampled_root / f"frame_{source_index:06d}.jpg"
        if not sampled_frame.is_file():
            raise RuntimeError(f"{label} is missing sampled frame {sampled_frame.name}")
        if not files_are_identical(chunk_frame, sampled_frame, digest_cache):
            raise RuntimeError(
                f"{label}/{chunk_frame.name} differs from sampled frame "
                f"{sampled_frame.name}"
            )


def validate_manifest_sources(repo_root: Path, manifest: dict[str, Any]) -> None:
    for video in manifest["videos"]:
        source = resolve_repo_path(repo_root, video["source"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing manifest source video: {source}")
        actual_digest = sha256(source)
        if actual_digest != video["source_sha256"]:
            raise RuntimeError(
                f"Source SHA-256 differs from the batch manifest for {video['id']}"
            )
        print(f"Verified source video {video['id']}: {actual_digest}", flush=True)


def summary_is_valid(summary: dict[str, Any], frame_count: int) -> bool:
    expected = {
        "trajectory": [1, frame_count, frame_count, 280, 504, 3],
        "camera_pose": [1, frame_count, 3, 4],
        "intrinsics": [1, frame_count, 3, 3],
        "pts3d_dynamic_score": [1, frame_count, 280, 504],
    }
    return summary.get("input_images") == frame_count and all(
        summary.get(key, {}).get("shape") == shape
        and summary.get(key, {}).get("finite_fraction") == 1.0
        for key, shape in expected.items()
    )


def recover_or_skip(
    repo_root: Path,
    manifest: dict[str, Any],
    video: dict[str, Any],
    chunk: dict[str, Any],
    frame_count: int,
) -> bool:
    output_dir = resolve_repo_path(repo_root, chunk["output_dir"])
    final_pt = resolve_repo_path(repo_root, chunk["prediction_file"])
    summary_path = output_dir / "prediction_summary.json"
    status_path = output_dir / "status.json"
    fingerprint = inference_fingerprint(manifest, video, chunk)
    label = f"{video['id']}/{chunk['id']}"

    if (
        not final_pt.is_file()
        or not summary_path.is_file()
        or not status_path.is_file()
    ):
        return False
    try:
        existing_status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"RECOVERY REJECT {label}: cannot read status " f"({type(error).__name__})",
            flush=True,
        )
        return False
    if not isinstance(existing_status, dict):
        print(f"RECOVERY REJECT {label}: status root is not an object", flush=True)
        return False
    if existing_status.get("state") != "complete":
        print(
            f"RECOVERY REJECT {label}: status is not complete; inference required",
            flush=True,
        )
        return False
    if existing_status.get("prediction_file") != chunk["prediction_file"]:
        print(
            f"RECOVERY REJECT {label}: canonical prediction path differs",
            flush=True,
        )
        return False

    status_fingerprint = existing_status.get("inference_fingerprint")
    is_legacy_status = status_fingerprint in (None, "")
    if not is_legacy_status and status_fingerprint != fingerprint:
        print(
            f"RECOVERY REJECT {label}: nonempty inference fingerprint differs",
            flush=True,
        )
        return False
    actual_bytes = final_pt.stat().st_size
    if existing_status.get("pt_bytes") != actual_bytes:
        print(f"RECOVERY REJECT {label}: PT byte size differs", flush=True)
        return False
    status_sha256 = existing_status.get("pt_sha256")
    if not isinstance(status_sha256, str) or not status_sha256:
        print(f"RECOVERY REJECT {label}: PT checksum is missing", flush=True)
        return False
    actual_sha256 = sha256(final_pt)
    if status_sha256 != actual_sha256:
        print(f"RECOVERY REJECT {label}: PT checksum differs", flush=True)
        return False

    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"RECOVERY REJECT {label}: cannot read summary "
            f"({type(error).__name__})",
            flush=True,
        )
        return False
    if not isinstance(summary, dict):
        print(f"RECOVERY REJECT {label}: summary root is not an object", flush=True)
        return False
    if not summary_is_valid(summary, frame_count):
        print(f"RECOVERY REJECT {label}: prediction summary is invalid", flush=True)
        return False
    batch_chunk_summary = summary.get("batch_chunk")
    summary_fingerprint = (
        batch_chunk_summary.get("inference_fingerprint")
        if isinstance(batch_chunk_summary, dict)
        else None
    )
    if summary_fingerprint and summary_fingerprint != fingerprint:
        print(f"RECOVERY REJECT {label}: summary fingerprint differs", flush=True)
        return False

    try:
        raw = torch.load(final_pt, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        print(
            f"RECOVERY REJECT {label}: cannot load PT ({type(error).__name__})",
            flush=True,
        )
        return False
    raw_shapes = expected_raw_shapes(frame_count)
    if not isinstance(raw, Mapping):
        print(f"RECOVERY REJECT {label}: raw PT root is not a mapping", flush=True)
        return False
    if any(
        key not in raw
        or not isinstance(raw[key], torch.Tensor)
        or list(raw[key].shape) != shape
        or raw[key].dtype != torch.float32
        for key, shape in raw_shapes.items()
    ):
        print(f"RECOVERY REJECT {label}: raw tensor schema differs", flush=True)
        return False

    if is_legacy_status:
        atomic_json(
            status_path,
            {
                **existing_status,
                "inference_fingerprint": fingerprint,
                "legacy_migration": {
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                    "rule": "complete-exact-path-size-sha256-and-raw-schema",
                },
            },
        )
    return True


def run(cfg: DictConfig) -> None:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    repo_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    manifest_path = resolve_repo_path(repo_root, cfg.paths.batch_manifest)
    checkpoint_path = resolve_repo_path(repo_root, cfg.paths.checkpoint_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "omnix.batch-plan.v1":
        raise ValueError(
            f"Unsupported batch manifest schema: {manifest.get('schema')!r}"
        )
    provenance = manifest["provenance"]
    checkpoint_relative = checkpoint_path.relative_to(repo_root).as_posix()
    if checkpoint_relative != provenance["checkpoint"]:
        raise RuntimeError(
            f"Checkpoint path {checkpoint_relative!r} differs from manifest "
            f"{provenance['checkpoint']!r}"
        )
    print("Verifying checkpoint and container provenance", flush=True)
    if sha256(checkpoint_path) != provenance["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA-256 differs from the batch manifest")
    actual_image_id = os.environ.get("OMNIX_IMAGE_ID")
    if actual_image_id and actual_image_id != provenance["docker_image_id"]:
        raise RuntimeError("Docker image ID differs from the batch manifest")
    frame_count = int(manifest["sampling"]["frames_per_chunk"])
    max_chunks = cfg.paths.get("max_chunks", None)
    max_chunks = int(max_chunks) if max_chunks is not None else None
    only_video = cfg.paths.get("only_video", None)
    only_chunk = cfg.paths.get("only_chunk", None)

    chunks = [
        (video, chunk)
        for video, chunk in flatten_chunks(manifest)
        if (only_video is None or video["id"] == str(only_video))
        and (only_chunk is None or chunk["id"] == str(only_chunk))
    ]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise RuntimeError("No manifest chunks matched the requested filters")

    print("Verifying manifest source videos", flush=True)
    validate_manifest_sources(repo_root, manifest)
    print(f"Validating inputs for {len(chunks)} selected chunks", flush=True)
    digest_cache: dict[Path, str] = {}
    for video, chunk in chunks:
        validate_chunk_inputs(repo_root, video, chunk, frame_count, digest_cache)

    print(f"Instantiating inference network <{cfg.model.net._target_}>", flush=True)
    model = hydra.utils.instantiate(cfg.model.net)
    load_inference_weights(model, checkpoint_path)
    model = model.to("cuda").eval()
    print(f"Ready to process {len(chunks)} chunks from {manifest_path}", flush=True)

    completed = 0
    for ordinal, (video, chunk) in enumerate(chunks, start=1):
        raw = predictions = batch = images = image_info = None
        video_id = video["id"]
        fingerprint = inference_fingerprint(manifest, video, chunk)
        label = f"{video_id}/{chunk['id']}"
        if recover_or_skip(repo_root, manifest, video, chunk, frame_count):
            completed += 1
            print(
                f"[{ordinal}/{len(chunks)}] SKIP {label}: validated output exists",
                flush=True,
            )
            continue

        input_dir = resolve_repo_path(repo_root, chunk["input_dir"])
        output_dir = resolve_repo_path(repo_root, chunk["output_dir"])
        final_pt = resolve_repo_path(repo_root, chunk["prediction_file"])
        summary_path = output_dir / "prediction_summary.json"
        status_path = output_dir / "status.json"
        partial_pt = final_pt.with_suffix(final_pt.suffix + ".partial")
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_pt.unlink(missing_ok=True)
        (output_dir / "predictions.pt").unlink(missing_ok=True)

        print(f"[{ordinal}/{len(chunks)}] RUN  {label}", flush=True)
        atomic_json(
            status_path,
            {
                "state": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "prediction_file": chunk["prediction_file"],
                "inference_fingerprint": fingerprint,
            },
        )
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            images, image_info = load_images_from_folder(
                input_dir, (504, 280), max_images=frame_count
            )
            if images.shape[0] != frame_count:
                raise RuntimeError(
                    f"{label} has {images.shape[0]} images, expected {frame_count}"
                )
            batch = {
                "image": images.unsqueeze(0),
                "image_info": image_info.unsqueeze(0),
            }
            with torch.inference_mode():
                predictions = model(batch)
            torch.cuda.synchronize()

            summary = summarize_predictions(predictions, REQUIRED_KEYS)
            summary["geometry_checks"] = validate_geometry(predictions, (280, 504))
            summary["input_images"] = int(images.shape[0])
            summary["batch_chunk"] = {
                "video_id": video_id,
                "chunk_id": chunk["id"],
                "valid_frames": chunk["valid_frames"],
                "pad_frames": chunk["pad_frames"],
                "start_time_s": chunk["start_time_s"],
                "end_time_s": chunk["end_time_s"],
                "inference_fingerprint": fingerprint,
            }
            if not summary_is_valid(summary, frame_count):
                raise RuntimeError(f"{label} failed summary shape/finite validation")
            atomic_json(summary_path, summary)

            raw = {
                key: predictions[key][0].detach().float().cpu() for key in REQUIRED_KEYS
            }
            torch.save(raw, partial_pt)
            os.replace(partial_pt, final_pt)
            elapsed = time.monotonic() - started
            atomic_json(
                status_path,
                {
                    "state": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": elapsed,
                    "prediction_file": chunk["prediction_file"],
                    "pt_bytes": final_pt.stat().st_size,
                    "pt_sha256": sha256(final_pt),
                    "inference_fingerprint": fingerprint,
                    "cuda_peak_memory_gib": summary["cuda_peak_memory_gib"],
                },
            )
            completed += 1
            print(
                f"[{ordinal}/{len(chunks)}] DONE {label} in {elapsed:.1f}s", flush=True
            )
        except Exception as error:
            atomic_json(
                status_path,
                {
                    "state": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "prediction_file": chunk["prediction_file"],
                    "inference_fingerprint": fingerprint,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            del raw, predictions, batch, images, image_info
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Batch complete: {completed}/{len(chunks)} chunks validated", flush=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    run(cfg)
    return None


if __name__ == "__main__":
    main()
