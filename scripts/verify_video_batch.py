#!/usr/bin/env python3
"""Verify a completed OmniX batch and write an aggregate acceptance report."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must stay inside the repository: {value}") from error
    return path


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


def range_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
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


def expected_raw_shapes(frame_count: int) -> dict[str, list[int]]:
    return {
        "trajectory": [frame_count, frame_count, 280, 504, 3],
        "camera_pose": [frame_count, 3, 4],
        "intrinsics": [frame_count, 3, 3],
        "pts3d_dynamic_score": [frame_count, 280, 504],
    }


def tensor_is_finite_chunked(
    tensor: torch.Tensor, maximum_elements: int = 16_000_000
) -> bool:
    if tensor.numel() <= maximum_elements:
        return bool(torch.isfinite(tensor).all().item())
    if tensor.ndim == 0:
        return bool(torch.isfinite(tensor).item())
    return all(
        tensor_is_finite_chunked(tensor[index], maximum_elements)
        for index in range(tensor.shape[0])
    )


def validate_raw_predictions(pt_path: Path, frame_count: int) -> list[str]:
    errors: list[str] = []
    try:
        raw = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        return [f"cannot mmap-load PT: {type(error).__name__}: {error}"]

    try:
        if not isinstance(raw, Mapping):
            return ["raw PT root is not a mapping"]
        for key, shape in expected_raw_shapes(frame_count).items():
            tensor = raw.get(key)
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"raw {key} tensor is missing")
                continue
            if list(tensor.shape) != shape:
                errors.append(f"raw {key}.shape={list(tensor.shape)} expected {shape}")
                continue
            if tensor.dtype != torch.float32:
                errors.append(f"raw {key}.dtype={tensor.dtype} expected torch.float32")
                continue
            try:
                if not tensor_is_finite_chunked(tensor):
                    errors.append(f"raw {key} contains non-finite values")
            except Exception as error:
                errors.append(
                    f"cannot check raw {key} finiteness: {type(error).__name__}: {error}"
                )
    finally:
        del raw
        gc.collect()
    return errors


def number_matches(value: Any, expected: float) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=5e-7)
    )


def validate_manifest_structure(
    manifest: dict[str, Any], frame_count: int, fps: float
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    def fail(scope: str, error: str) -> None:
        failures.append({"chunk": scope, "error": error})

    sampling = manifest["sampling"]
    if sampling.get("non_overlapping") is not True:
        fail("manifest", "sampling.non_overlapping must be true")
    if sampling.get("tail_policy") != "repeat-last-frame-padding":
        fail("manifest", "unsupported tail padding policy")

    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        fail("manifest", "videos must be a nonempty list")
        return failures
    video_ids = [video.get("id") for video in videos if isinstance(video, dict)]
    if len(video_ids) != len(videos) or any(
        not isinstance(video_id, str) or not video_id for video_id in video_ids
    ):
        fail("manifest", "every video must have a nonempty string ID")
    if len(video_ids) != len(set(video_ids)):
        fail("manifest", "video IDs must be unique")

    actual_total_chunks = 0
    for video in videos:
        if not isinstance(video, dict):
            continue
        video_id = str(video.get("id", "<missing-video-id>"))
        chunks = video.get("chunks")
        if not isinstance(chunks, list):
            fail(video_id, "chunks must be a list")
            continue
        actual_total_chunks += len(chunks)
        declared_chunk_count = video.get("chunk_count")
        if type(declared_chunk_count) is not int or declared_chunk_count != len(chunks):
            fail(
                video_id,
                f"chunk_count={declared_chunk_count!r} differs from {len(chunks)} entries",
            )
        sampled_frame_count = video.get("sampled_frame_count")
        if type(sampled_frame_count) is not int or sampled_frame_count <= 0:
            fail(video_id, "sampled_frame_count must be a positive integer")
            sampled_frame_count = 0
        expected_chunk_count = (
            math.ceil(sampled_frame_count / frame_count)
            if sampled_frame_count > 0
            else 0
        )
        if len(chunks) != expected_chunk_count:
            fail(
                video_id,
                f"chunk list has {len(chunks)} entries; coverage requires "
                f"{expected_chunk_count}",
            )

        expected_start = 0
        for ordinal, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                fail(video_id, f"chunk entry {ordinal} is not an object")
                continue
            expected_id = f"chunk_{ordinal:04d}"
            chunk_id = str(chunk.get("id", f"<chunk-{ordinal}>"))
            label = f"{video_id}/{chunk_id}"
            if chunk.get("index") != ordinal:
                fail(label, f"index={chunk.get('index')!r} expected {ordinal}")
            if chunk_id != expected_id:
                fail(label, f"chunk ID must be {expected_id}")

            valid_frames = chunk.get("valid_frames")
            pad_frames = chunk.get("pad_frames")
            valid_is_int = type(valid_frames) is int
            pad_is_int = type(pad_frames) is int
            if not valid_is_int or not 1 <= valid_frames <= frame_count:
                fail(
                    label,
                    f"valid_frames={valid_frames!r} is outside [1, {frame_count}]",
                )
            if not pad_is_int or not 0 <= pad_frames < frame_count:
                fail(label, f"pad_frames={pad_frames!r} is outside [0, {frame_count})")
            if valid_is_int and pad_is_int and valid_frames + pad_frames != frame_count:
                fail(label, "valid_frames + pad_frames differs from frames_per_chunk")
            if ordinal < len(chunks) - 1 and (
                valid_frames != frame_count or pad_frames != 0
            ):
                fail(label, "padding is only permitted in the final chunk")

            if not valid_is_int or valid_frames <= 0:
                continue
            start = chunk.get("sample_start_index")
            end = chunk.get("sample_end_index_exclusive")
            expected_end = expected_start + valid_frames
            if start != expected_start:
                fail(label, f"sample_start_index={start!r} expected {expected_start}")
            if end != expected_end:
                fail(
                    label,
                    f"sample_end_index_exclusive={end!r} expected {expected_end}",
                )

            source_indices = chunk.get("source_sample_indices")
            expected_indices = list(range(expected_start, expected_end))
            if pad_is_int and pad_frames > 0:
                expected_indices.extend([expected_end - 1] * pad_frames)
            if (
                not isinstance(source_indices, list)
                or any(type(index) is not int for index in source_indices)
                or source_indices != expected_indices
            ):
                fail(
                    label,
                    "source_sample_indices are not contiguous with repeat-last tail padding",
                )

            expected_times = [round(index / fps, 6) for index in expected_indices]
            sample_times = chunk.get("sample_times_s")
            if (
                not isinstance(sample_times, list)
                or len(sample_times) != len(expected_times)
                or any(
                    not number_matches(value, expected)
                    for value, expected in zip(sample_times, expected_times)
                )
            ):
                fail(
                    label, "sample_times_s differ from source indices and sampling FPS"
                )
            if not number_matches(
                chunk.get("start_time_s"), round(expected_start / fps, 6)
            ):
                fail(label, "start_time_s differs from sample_start_index")
            if not number_matches(
                chunk.get("end_time_s"), round(expected_end / fps, 6)
            ):
                fail(label, "end_time_s differs from sample_end_index_exclusive")
            expected_start = expected_end

        if expected_start != sampled_frame_count:
            fail(
                video_id,
                f"chunk coverage ends at {expected_start}, expected "
                f"{sampled_frame_count}",
            )
        if chunks and sampled_frame_count > 0:
            expected_tail_padding = (
                frame_count - sampled_frame_count % frame_count
            ) % frame_count
            if chunks[-1].get("pad_frames") != expected_tail_padding:
                fail(
                    f"{video_id}/{chunks[-1].get('id', '<last>')}",
                    f"tail padding differs from expected {expected_tail_padding}",
                )

    declared_total_chunks = manifest.get("total_chunks")
    if (
        type(declared_total_chunks) is not int
        or declared_total_chunks != actual_total_chunks
    ):
        fail(
            "manifest",
            f"total_chunks={declared_total_chunks!r} differs from "
            f"{actual_total_chunks} chunk entries",
        )
    return failures


def validate_summary(summary: dict[str, Any], frame_count: int) -> list[str]:
    errors: list[str] = []
    expected = {
        "trajectory": [1, frame_count, frame_count, 280, 504, 3],
        "camera_pose": [1, frame_count, 3, 4],
        "intrinsics": [1, frame_count, 3, 3],
        "pts3d_dynamic_score": [1, frame_count, 280, 504],
    }
    if summary.get("input_images") != frame_count:
        errors.append(
            f"input_images={summary.get('input_images')} expected {frame_count}"
        )
    for key, shape in expected.items():
        value = summary.get(key, {})
        if not isinstance(value, dict):
            errors.append(f"{key} summary is not an object")
            continue
        if value.get("shape") != shape:
            errors.append(f"{key}.shape={value.get('shape')} expected {shape}")
        if value.get("finite_fraction") != 1.0:
            errors.append(f"{key}.finite_fraction={value.get('finite_fraction')}")

    geometry = summary.get("geometry_checks", {})
    if not isinstance(geometry, dict):
        errors.append("geometry_checks is not an object")
        return errors
    if geometry.get("rotation_orthogonality_max_error", float("inf")) > 1e-3:
        errors.append("camera rotation orthogonality exceeds 1e-3")
    if geometry.get("rotation_determinant_min", 0) < 0.99:
        errors.append("camera rotation determinant minimum is below 0.99")
    if geometry.get("rotation_determinant_max", 2) > 1.01:
        errors.append("camera rotation determinant maximum is above 1.01")
    if (
        geometry.get("focal_length_x_min", 0) <= 0
        or geometry.get("focal_length_y_min", 0) <= 0
    ):
        errors.append("predicted focal length is not positive")
    dynamic_fraction = geometry.get("dynamic_fraction_above_0_5", -1)
    if not 0 <= dynamic_fraction <= 1:
        errors.append("dynamic fraction is outside [0, 1]")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--rehash", action="store_true", help="Recompute every PT SHA-256"
    )
    args = parser.parse_args()

    manifest_path = repo_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "omnix.batch-plan.v1":
        raise ValueError(
            f"Unsupported batch manifest schema: {manifest.get('schema')!r}"
        )
    output_root = repo_path(manifest["output_root"])
    frame_count = int(manifest["sampling"]["frames_per_chunk"])
    fps = float(manifest["sampling"]["fps"])
    if frame_count <= 0 or fps <= 0 or not math.isfinite(fps):
        raise ValueError("Manifest frames_per_chunk and FPS must be positive")
    failures = validate_manifest_structure(manifest, frame_count, fps)
    video_reports = []
    total_chunks_to_check = sum(len(video["chunks"]) for video in manifest["videos"])
    checked_ordinal = 0

    for video in manifest["videos"]:
        pt_bytes: list[int] = []
        peak_memory: list[float] = []
        dynamic_fraction: list[float] = []
        temporal_delta_mean: list[float] = []
        runtimes: list[float] = []
        valid_total = 0
        pad_total = 0
        completed = 0

        for chunk in video["chunks"]:
            checked_ordinal += 1
            label = f"{video['id']}/{chunk['id']}"
            print(
                f"[{checked_ordinal}/{total_chunks_to_check}] CHECK {label}",
                flush=True,
            )
            pt_path = repo_path(chunk["prediction_file"])
            output_dir = repo_path(chunk["output_dir"])
            status_path = output_dir / "status.json"
            summary_path = output_dir / "prediction_summary.json"
            chunk_errors: list[str] = []
            status: dict[str, Any] | None = None
            summary: dict[str, Any] | None = None

            if not pt_path.is_file():
                chunk_errors.append("prediction PT is missing")
            else:
                chunk_errors.extend(validate_raw_predictions(pt_path, frame_count))
            if status_path.is_file():
                try:
                    status_value = json.loads(status_path.read_text())
                    if isinstance(status_value, dict):
                        status = status_value
                    else:
                        chunk_errors.append("status.json root is not an object")
                except (OSError, json.JSONDecodeError) as error:
                    chunk_errors.append(
                        f"cannot read status.json: {type(error).__name__}: {error}"
                    )
            else:
                chunk_errors.append("status.json is missing")
            if summary_path.is_file():
                try:
                    summary_value = json.loads(summary_path.read_text())
                    if isinstance(summary_value, dict):
                        summary = summary_value
                    else:
                        chunk_errors.append(
                            "prediction_summary.json root is not an object"
                        )
                except (OSError, json.JSONDecodeError) as error:
                    chunk_errors.append(
                        "cannot read prediction_summary.json: "
                        f"{type(error).__name__}: {error}"
                    )
            else:
                chunk_errors.append("prediction_summary.json is missing")

            if status is not None:
                if status.get("state") != "complete":
                    chunk_errors.append(f"status state is {status.get('state')!r}")
                if status.get("prediction_file") != chunk["prediction_file"]:
                    chunk_errors.append(
                        "status prediction_file differs from the canonical path"
                    )
                if (
                    pt_path.is_file()
                    and status.get("pt_bytes") != pt_path.stat().st_size
                ):
                    chunk_errors.append("PT byte size does not match status")
                if status.get("inference_fingerprint") != inference_fingerprint(
                    manifest, video, chunk
                ):
                    chunk_errors.append("inference fingerprint does not match manifest")
                status_sha256 = status.get("pt_sha256")
                if not isinstance(status_sha256, str) or not status_sha256:
                    chunk_errors.append("PT SHA-256 is missing from status")
                elif (
                    args.rehash
                    and pt_path.is_file()
                    and status_sha256 != sha256(pt_path)
                ):
                    chunk_errors.append("PT SHA-256 does not match status")
            if summary is not None:
                chunk_errors.extend(validate_summary(summary, frame_count))
                batch_chunk = summary.get("batch_chunk")
                if isinstance(batch_chunk, dict):
                    summary_fingerprint = batch_chunk.get("inference_fingerprint")
                    if (
                        summary_fingerprint
                        and summary_fingerprint
                        != inference_fingerprint(manifest, video, chunk)
                    ):
                        chunk_errors.append(
                            "prediction summary fingerprint does not match manifest"
                        )
            if chunk_errors:
                failures.extend(
                    {"chunk": label, "error": error} for error in chunk_errors
                )
                continue

            assert status is not None and summary is not None
            completed += 1
            valid_total += int(chunk["valid_frames"])
            pad_total += int(chunk["pad_frames"])
            pt_bytes.append(pt_path.stat().st_size)
            peak_memory.append(float(summary["cuda_peak_memory_gib"]))
            geometry = summary["geometry_checks"]
            dynamic_fraction.append(float(geometry["dynamic_fraction_above_0_5"]))
            temporal_delta_mean.append(float(geometry["temporal_delta_mean"]))
            if status.get("runtime_seconds") is not None:
                runtimes.append(float(status["runtime_seconds"]))

        expected_video_chunks = len(video["chunks"])
        if completed != expected_video_chunks:
            failures.append(
                {
                    "chunk": video["id"],
                    "error": f"completed {completed} chunks; expected "
                    f"{expected_video_chunks}",
                }
            )
        else:
            if valid_total != video["sampled_frame_count"]:
                failures.append(
                    {
                        "chunk": video["id"],
                        "error": "valid frame coverage differs from sampled frame count",
                    }
                )
            expected_padding = (
                expected_video_chunks * frame_count - video["sampled_frame_count"]
            )
            if pad_total != expected_padding:
                failures.append(
                    {
                        "chunk": video["id"],
                        "error": "padding frame count differs from fixed-shape requirement",
                    }
                )

        video_reports.append(
            {
                "id": video["id"],
                "expected_chunks": expected_video_chunks,
                "completed_chunks": completed,
                "sampled_frames": video["sampled_frame_count"],
                "valid_frames_covered": valid_total,
                "padding_frames": pad_total,
                "pt_bytes": sum(pt_bytes),
                "peak_cuda_memory_gib": (
                    range_stats(peak_memory) if peak_memory else None
                ),
                "dynamic_fraction_above_0_5": (
                    range_stats(dynamic_fraction) if dynamic_fraction else None
                ),
                "temporal_delta_mean": (
                    range_stats(temporal_delta_mean) if temporal_delta_mean else None
                ),
                "runtime_seconds": {
                    **(range_stats(runtimes) if runtimes else {}),
                    "total": sum(runtimes),
                    "recorded_chunks": len(runtimes),
                },
            }
        )

    completed_total = sum(video["completed_chunks"] for video in video_reports)
    if completed_total != total_chunks_to_check:
        failures.append(
            {
                "chunk": "manifest",
                "error": f"completed {completed_total} total chunks; expected "
                f"{total_chunks_to_check}",
            }
        )
    report = {
        "schema": "omnix.batch-report.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": manifest_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "rehash_performed": args.rehash,
        "accepted": not failures,
        "expected_chunks": manifest["total_chunks"],
        "completed_chunks": completed_total,
        "total_pt_bytes": sum(video["pt_bytes"] for video in video_reports),
        "videos": video_reports,
        "failures": failures,
    }
    report_path = output_root / "batch_report.json"
    atomic_json(report_path, report)
    print(
        f"Validated {report['completed_chunks']}/{report['expected_chunks']} chunks; "
        f"accepted={report['accepted']}; wrote {report_path}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
