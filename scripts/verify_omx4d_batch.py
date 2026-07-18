#!/usr/bin/env python3
"""Exhaustively verify a 500k-point OMX4D batch and publish its catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"OMX4D\r\n\x1a"
SCHEMA_VERSION = 1
PREFIX = struct.Struct("<8sII")
ALIGNMENT = 8
MAX_HEADER_BYTES = 1024 * 1024
POINT_COUNT = 500_000
DYNAMIC_FRACTION = 0.8
DYNAMIC_COUNT = 400_000
SPATIAL_COUNT = 100_000
DYNAMIC_THRESHOLD = 0.0
SECTION_ORDER = (
    "positions",
    "colors",
    "dynamicScore",
    "sourceView",
    "cameraPose",
    "intrinsics",
)
SECTION_DTYPES = {
    "positions": ("float32", np.dtype("<f4")),
    "colors": ("uint8", np.dtype("u1")),
    "dynamicScore": ("float32", np.dtype("<f4")),
    "sourceView": ("uint16", np.dtype("<u2")),
    "cameraPose": ("float32", np.dtype("<f4")),
    "intrinsics": ("float32", np.dtype("<f4")),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object contains an ambiguous duplicate key."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary followed by os.replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKey) as error:
        raise ValueError(f"{label} is not valid unambiguous UTF-8 JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return parsed


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes(), str(path))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def sha256(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def is_int(value: Any) -> bool:
    return type(value) is int


def number_matches(value: Any, expected: float, tolerance: float = 1e-9) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and math.isclose(
            float(value), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    )


def safe_repo_path(repo_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {value}") from error
    return path


def repo_relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def inference_fingerprint(
    plan: Mapping[str, Any],
    video: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> str:
    """Reproduce the fingerprint bound into each source inference status."""

    payload = {
        "schema": plan.get("schema"),
        "sampling": plan.get("sampling"),
        "provenance": plan.get("provenance"),
        "video": {
            "id": video.get("id"),
            "source_sha256": video.get("source_sha256"),
            "crop_filter": video.get("crop_filter"),
        },
        "chunk": {
            "id": chunk.get("id"),
            "source_sample_indices": chunk.get("source_sample_indices"),
            "valid_frames": chunk.get("valid_frames"),
            "pad_frames": chunk.get("pad_frames"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be a nonempty path-safe name")
    if value in (".", ".."):
        raise ValueError(f"{label} must not be {value!r}")
    return value


def add_error(
    errors: list[dict[str, str]], scope: str, path: Path | str, message: str
) -> None:
    errors.append({"chunk": scope, "path": str(path), "error": message})


def read_omx4d_manifest(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        prefix = handle.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError("OMX4D prefix is truncated")
        magic, version, header_length = PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError("OMX4D magic is invalid")
        if version != SCHEMA_VERSION:
            raise ValueError(f"OMX4D envelope version is {version}, expected 1")
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise ValueError(
                f"OMX4D header length {header_length} is outside the accepted range"
            )
        header = handle.read(header_length)
        if len(header) != header_length:
            raise ValueError("OMX4D JSON header is truncated")
    return parse_json_bytes(header, f"{path} header"), header_length


def expected_section_shapes(
    frame_count: int,
) -> dict[str, list[int]]:
    return {
        "positions": [frame_count, POINT_COUNT, 3],
        "colors": [POINT_COUNT, 3],
        "dynamicScore": [POINT_COUNT],
        "sourceView": [POINT_COUNT],
        "cameraPose": [frame_count, 4, 4],
        "intrinsics": [frame_count, 3, 3],
    }


def validate_envelope_contract(
    path: Path,
    manifest: Mapping[str, Any],
    header_length: int,
    frame_count: int,
) -> tuple[dict[str, dict[str, Any]], int, list[str]]:
    """Validate exact descriptors and return safe descriptors plus expected size."""

    failures: list[str] = []
    attributes = manifest.get("attributes")
    if not isinstance(attributes, dict):
        return {}, 0, ["manifest attributes must be an object"]
    actual_names = set(attributes)
    expected_names = set(SECTION_ORDER)
    if actual_names != expected_names:
        failures.append(
            "attributes must contain exactly the six sections; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    shapes = expected_section_shapes(frame_count)
    cursor = align(PREFIX.size + header_length)
    safe_descriptors: dict[str, dict[str, Any]] = {}
    for name in SECTION_ORDER:
        descriptor = attributes.get(name)
        if not isinstance(descriptor, dict):
            failures.append(f"attributes.{name} must be an object")
            continue
        expected_dtype_name, numpy_dtype = SECTION_DTYPES[name]
        expected_shape = shapes[name]
        expected_bytes = math.prod(expected_shape) * numpy_dtype.itemsize
        expected_offset = align(cursor)

        if set(descriptor) != {"offset", "byteLength", "dtype", "shape"}:
            failures.append(
                f"attributes.{name} must contain exactly offset/byteLength/dtype/shape"
            )
        offset = descriptor.get("offset")
        byte_length = descriptor.get("byteLength")
        shape = descriptor.get("shape")
        if not is_int(offset) or offset < 0:
            failures.append(f"attributes.{name}.offset must be a nonnegative integer")
        elif offset % ALIGNMENT != 0:
            failures.append(f"attributes.{name}.offset is not {ALIGNMENT}-byte aligned")
        elif offset != expected_offset:
            failures.append(
                f"attributes.{name}.offset={offset}, expected contiguous aligned offset "
                f"{expected_offset}"
            )
        if not is_int(byte_length) or byte_length != expected_bytes:
            failures.append(
                f"attributes.{name}.byteLength={byte_length!r}, expected {expected_bytes}"
            )
        if descriptor.get("dtype") != expected_dtype_name:
            failures.append(
                f"attributes.{name}.dtype={descriptor.get('dtype')!r}, "
                f"expected {expected_dtype_name!r}"
            )
        if (
            not isinstance(shape, list)
            or any(not is_int(dimension) for dimension in shape)
            or shape != expected_shape
        ):
            failures.append(
                f"attributes.{name}.shape={shape!r}, expected {expected_shape}"
            )

        if (
            is_int(offset)
            and offset == expected_offset
            and is_int(byte_length)
            and byte_length == expected_bytes
            and descriptor.get("dtype") == expected_dtype_name
            and shape == expected_shape
        ):
            safe_descriptors[name] = dict(descriptor)
        cursor = expected_offset + expected_bytes

    file_size = path.stat().st_size
    if file_size != cursor:
        failures.append(f"file size={file_size}, expected descriptor end={cursor}")
    if len(safe_descriptors) != len(SECTION_ORDER):
        failures.append("one or more sections are unsafe to memory-map")
    return safe_descriptors, cursor, failures


def section_memmap(
    path: Path, descriptor: Mapping[str, Any], name: str
) -> np.memmap:
    _, dtype = SECTION_DTYPES[name]
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=int(descriptor["offset"]),
        shape=tuple(int(value) for value in descriptor["shape"]),
        order="C",
    )


def scan_positions(
    path: Path, descriptor: Mapping[str, Any], vectors_per_block: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray]:
    positions = section_memmap(path, descriptor, "positions")
    vectors = positions.reshape(-1, 3)
    minimum = np.full(3, np.inf, dtype=np.float32)
    maximum = np.full(3, -np.inf, dtype=np.float32)
    try:
        for start in range(0, vectors.shape[0], vectors_per_block):
            block = np.asarray(vectors[start : start + vectors_per_block])
            if not bool(np.isfinite(block).all()):
                raise ValueError("positions contains NaN or infinite values")
            minimum = np.minimum(minimum, block.min(axis=0))
            maximum = np.maximum(maximum, block.max(axis=0))
    finally:
        del vectors
        del positions
    return minimum, maximum


def scan_dynamic_score(
    path: Path, descriptor: Mapping[str, Any], elements_per_block: int = 4_000_000
) -> tuple[float, float]:
    scores = section_memmap(path, descriptor, "dynamicScore").reshape(-1)
    minimum = math.inf
    maximum = -math.inf
    try:
        for start in range(0, scores.size, elements_per_block):
            block = np.asarray(scores[start : start + elements_per_block])
            if not bool(np.isfinite(block).all()):
                raise ValueError("dynamicScore contains NaN or infinite values")
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
    finally:
        del scores
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"dynamicScore range [{minimum}, {maximum}] is outside [0, 1]"
        )
    return minimum, maximum


def validate_small_sections(
    path: Path,
    descriptors: Mapping[str, Mapping[str, Any]],
    valid_frames: int,
) -> list[str]:
    failures: list[str] = []
    source_views = section_memmap(path, descriptors["sourceView"], "sourceView")
    try:
        source_min = int(source_views.min())
        source_max = int(source_views.max())
        if source_min < 0 or source_max >= valid_frames:
            failures.append(
                f"sourceView range [{source_min}, {source_max}] includes padded "
                f"source views; expected values below {valid_frames}"
            )
    finally:
        del source_views

    for name in ("cameraPose", "intrinsics"):
        calibration = section_memmap(path, descriptors[name], name)
        try:
            if not bool(np.isfinite(calibration).all()):
                failures.append(f"{name} contains NaN or infinite values")
        finally:
            del calibration
    return failures


def parsed_bounds(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        return None
    minimum = value.get("min")
    maximum = value.get("max")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
        or any(type(item) not in (int, float) for item in minimum + maximum)
    ):
        return None
    arrays = (
        np.asarray(minimum, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
    )
    if not all(bool(np.isfinite(array).all()) for array in arrays):
        return None
    return arrays


def validate_manifest_values(
    manifest: Mapping[str, Any],
    fps: float,
    frame_count: int,
    valid_frames: int,
    width: int,
    height: int,
) -> list[str]:
    failures: list[str] = []
    exact_values = {
        "schemaVersion": SCHEMA_VERSION,
        "frameCount": frame_count,
        "sourceViewCount": frame_count,
        "pointCount": POINT_COUNT,
        "primitive": "points",
    }
    for field, expected in exact_values.items():
        actual = manifest.get(field)
        if (
            (type(expected) is int and (not is_int(actual) or actual != expected))
            or (type(expected) is str and actual != expected)
        ):
            failures.append(f"manifest {field}={manifest.get(field)!r}, expected {expected!r}")
    if not number_matches(manifest.get("fps"), fps):
        failures.append(f"manifest fps={manifest.get('fps')!r}, expected {fps}")
    if not number_matches(manifest.get("durationSeconds"), frame_count / fps):
        failures.append(
            f"manifest durationSeconds={manifest.get('durationSeconds')!r}, "
            f"expected {frame_count / fps}"
        )

    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        return failures + ["manifest sampling must be an object"]
    sampling_exact = {
        "method": "dynamic-reserved-voxel-v1",
        "requestedPointCount": POINT_COUNT,
        "selectedPointCount": POINT_COUNT,
        "dynamicSelectedPointCount": DYNAMIC_COUNT,
        "spatialSelectedPointCount": SPATIAL_COUNT,
        "candidateSourceViewCount": valid_frames,
        "excludedPaddedSourceViewCount": frame_count - valid_frames,
        "validCandidateCount": valid_frames * height * width,
        "dynamicRanking": "global-descending-stable-identity-tiebreak",
        "spatialDistribution": "normalized-frame-zero-3d-voxel",
    }
    for field, expected in sampling_exact.items():
        actual = sampling.get(field)
        if (
            (type(expected) is int and (not is_int(actual) or actual != expected))
            or (type(expected) is str and actual != expected)
        ):
            failures.append(
                f"sampling.{field}={sampling.get(field)!r}, expected {expected!r}"
            )
    if not number_matches(
        sampling.get("dynamicThreshold"), DYNAMIC_THRESHOLD, tolerance=0.0
    ):
        failures.append(
            f"sampling.dynamicThreshold={sampling.get('dynamicThreshold')!r}, "
            f"expected exactly {DYNAMIC_THRESHOLD}"
        )
    if not number_matches(
        sampling.get("dynamicReservedFraction"), DYNAMIC_FRACTION, tolerance=0.0
    ):
        failures.append(
            "sampling.dynamicReservedFraction="
            f"{sampling.get('dynamicReservedFraction')!r}, expected exactly "
            f"{DYNAMIC_FRACTION}"
        )
    if not is_sha256(sampling.get("identityHash")):
        failures.append("sampling.identityHash must be a lowercase SHA-256")
    cutoff = sampling.get("dynamicScoreCutoff")
    if (
        type(cutoff) not in (int, float)
        or not math.isfinite(float(cutoff))
        or not 0.0 <= float(cutoff) <= 1.0
    ):
        failures.append("sampling.dynamicScoreCutoff must be finite and within [0, 1]")

    warnings = manifest.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(warning, str) for warning in warnings
    ):
        failures.append("manifest warnings must be a list of strings")
    elif any("Source RGB was not supplied" in warning for warning in warnings):
        failures.append("manifest indicates palette fallback instead of source RGB")
    return failures


def compare_bounds(
    label: str,
    value: Any,
    actual_minimum: np.ndarray,
    actual_maximum: np.ndarray,
) -> list[str]:
    parsed = parsed_bounds(value)
    if parsed is None:
        return [f"{label} bounds must contain finite three-vector min/max values"]
    minimum, maximum = parsed
    failures: list[str] = []
    if not np.array_equal(minimum, actual_minimum):
        failures.append(
            f"{label} bounds.min={minimum.tolist()} differs from actual "
            f"{actual_minimum.tolist()}"
        )
    if not np.array_equal(maximum, actual_maximum):
        failures.append(
            f"{label} bounds.max={maximum.tolist()} differs from actual "
            f"{actual_maximum.tolist()}"
        )
    return failures


def validate_sidecar_and_status(
    *,
    repo_root: Path,
    output_path: Path,
    sidecar: Mapping[str, Any],
    status_path: Path,
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_sha256: str,
    plan: Mapping[str, Any],
    plan_video: Mapping[str, Any],
    plan_chunk: Mapping[str, Any],
    fps: float,
    frame_count: int,
    width: int,
    height: int,
) -> list[str]:
    failures: list[str] = []
    video_id = str(plan_video["id"])
    chunk_id = str(plan_chunk["id"])
    prediction_value = str(plan_chunk["prediction_file"])
    input_value = str(plan_chunk["input_dir"])
    output_value = repo_relative(repo_root, output_path)
    prediction_path = safe_repo_path(repo_root, prediction_value)

    sidecar_exact = {
        "schema": "omnix.omx4d-bake.v1",
        "video": video_id,
        "chunk": chunk_id,
        "valid_frames": int(plan_chunk["valid_frames"]),
        "pad_frames": int(plan_chunk["pad_frames"]),
        "source_pt": prediction_value,
        "source_rgb": input_value,
        "output": output_value,
        "bytes": output_path.stat().st_size,
        "sha256": output_sha256,
        "point_count": POINT_COUNT,
        "frame_count": frame_count,
        "source_resolution": [width, height],
    }
    for field, expected in sidecar_exact.items():
        actual = sidecar.get(field)
        exact = actual == expected
        if type(expected) is int:
            exact = is_int(actual) and actual == expected
        if not exact:
            failures.append(
                f"sidecar {field}={sidecar.get(field)!r}, expected {expected!r}"
            )
    for field in ("start_time_s", "end_time_s"):
        if not number_matches(sidecar.get(field), float(plan_chunk[field])):
            failures.append(
                f"sidecar {field}={sidecar.get(field)!r}, expected {plan_chunk[field]!r}"
            )
    if not number_matches(sidecar.get("fps"), fps):
        failures.append(f"sidecar fps={sidecar.get('fps')!r}, expected {fps}")
    if sidecar.get("sampling") != manifest.get("sampling"):
        failures.append("sidecar sampling does not exactly match the OMX4D manifest")
    if sidecar.get("bounds") != manifest.get("bounds"):
        failures.append("sidecar bounds do not exactly match the OMX4D manifest")
    if sidecar.get("warnings") != manifest.get("warnings"):
        failures.append("sidecar warnings do not exactly match the OMX4D manifest")

    if status.get("state") != "complete":
        failures.append(f"source status state={status.get('state')!r}, expected 'complete'")
    if status.get("prediction_file") != prediction_value:
        failures.append("source status prediction_file differs from the batch plan")
    if not prediction_path.is_file():
        failures.append(f"source prediction PT is missing: {prediction_path}")
    elif status.get("pt_bytes") != prediction_path.stat().st_size:
        failures.append("source status pt_bytes differs from the source PT size")
    source_sha = status.get("pt_sha256")
    if not is_sha256(source_sha):
        failures.append("source status pt_sha256 must be a lowercase SHA-256")
    if sidecar.get("source_pt_sha256") != source_sha:
        failures.append("sidecar source_pt_sha256 differs from source status")
    fingerprint = status.get("inference_fingerprint")
    if not is_sha256(fingerprint):
        failures.append("source status inference_fingerprint must be a lowercase SHA-256")
    expected_fingerprint = inference_fingerprint(plan, plan_video, plan_chunk)
    if fingerprint != expected_fingerprint:
        failures.append("source status inference_fingerprint differs from the batch plan")
    if sidecar.get("inference_fingerprint") != fingerprint:
        failures.append("sidecar inference_fingerprint differs from source status")

    expected_status = safe_repo_path(repo_root, str(plan_chunk["output_dir"])) / "status.json"
    if status_path != expected_status:
        failures.append("resolved source status path differs from plan output_dir/status.json")
    if prediction_path.parent != expected_status.parent:
        failures.append("prediction_file is not located in the plan output_dir")
    return failures


def plan_summary(
    plan: Mapping[str, Any], frame_count: int, repo_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    videos = plan.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("batch plan videos must be a nonempty list")
    normalized: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_video_ids: set[str] = set()
    for video_index, video_value in enumerate(videos):
        if not isinstance(video_value, dict):
            failures.append(f"video entry {video_index} is not an object")
            continue
        video = dict(video_value)
        try:
            video_id = safe_component(video.get("id"), f"video {video_index} id")
        except ValueError as error:
            failures.append(str(error))
            continue
        if video_id in seen_video_ids:
            failures.append(f"duplicate video ID {video_id!r}")
            continue
        seen_video_ids.add(video_id)
        chunks_value = video.get("chunks")
        if not isinstance(chunks_value, list) or not chunks_value:
            failures.append(f"{video_id} chunks must be a nonempty list")
            continue
        if video.get("chunk_count") != len(chunks_value):
            failures.append(
                f"{video_id} chunk_count={video.get('chunk_count')!r}, "
                f"expected {len(chunks_value)}"
            )
        chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for ordinal, chunk_value in enumerate(chunks_value):
            if not isinstance(chunk_value, dict):
                failures.append(f"{video_id} chunk entry {ordinal} is not an object")
                continue
            chunk = dict(chunk_value)
            try:
                chunk_id = safe_component(
                    chunk.get("id"), f"{video_id} chunk {ordinal} id"
                )
            except ValueError as error:
                failures.append(str(error))
                continue
            if chunk_id in seen_chunk_ids:
                failures.append(f"duplicate chunk ID {video_id}/{chunk_id}")
                continue
            seen_chunk_ids.add(chunk_id)
            valid_frames = chunk.get("valid_frames")
            pad_frames = chunk.get("pad_frames")
            if (
                not is_int(valid_frames)
                or not is_int(pad_frames)
                or not 1 <= valid_frames <= frame_count
                or not 0 <= pad_frames < frame_count
                or valid_frames + pad_frames != frame_count
            ):
                failures.append(
                    f"{video_id}/{chunk_id} valid_frames + pad_frames must be "
                    f"{frame_count}, with at least one valid frame"
                )
                continue
            required_paths = ("input_dir", "output_dir", "prediction_file")
            if any(
                not isinstance(chunk.get(field), str) or not chunk.get(field)
                for field in required_paths
            ):
                failures.append(
                    f"{video_id}/{chunk_id} is missing input/output/prediction paths"
                )
                continue
            try:
                for field in required_paths:
                    safe_repo_path(repo_root, str(chunk[field]))
            except ValueError as error:
                failures.append(f"{video_id}/{chunk_id} {error}")
                continue
            time_values = (chunk.get("start_time_s"), chunk.get("end_time_s"))
            if any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in time_values
            ):
                failures.append(f"{video_id}/{chunk_id} has invalid time bounds")
                continue
            chunks.append(chunk)
        video["chunks"] = chunks
        normalized.append(video)
    return normalized, failures


def expected_output_files(
    output_root: Path, videos: Sequence[Mapping[str, Any]]
) -> tuple[set[Path], set[Path]]:
    binaries: set[Path] = set()
    sidecars: set[Path] = set()
    for video in videos:
        video_id = str(video["id"])
        for chunk in video["chunks"]:
            stem = output_root / video_id / str(chunk["id"])
            binary = stem.with_suffix(".omx4d")
            sidecar = stem.with_suffix(".json")
            if binary in binaries or sidecar in sidecars:
                raise ValueError(f"duplicate expected output path for {stem}")
            binaries.add(binary)
            sidecars.add(sidecar)
    return binaries, sidecars


def actual_payload_files(output_root: Path) -> set[Path]:
    if not output_root.is_dir():
        return set()
    files: set[Path] = set()
    for path in output_root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        if path.parent == output_root and (
            path.name in {"catalog.json", "validation_report.json"}
            or (path.name.startswith("worker_summary_") and path.suffix == ".json")
        ):
            continue
        files.add(path)
    return files


def scalar_plan_fields(plan: Mapping[str, Any]) -> tuple[float, int, int, int]:
    if plan.get("schema") != "omnix.batch-plan.v1":
        raise ValueError(f"unsupported batch plan schema: {plan.get('schema')!r}")
    sampling = plan.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("batch plan sampling must be an object")
    fps = sampling.get("fps")
    frame_count = sampling.get("frames_per_chunk")
    width = sampling.get("model_width")
    height = sampling.get("model_height")
    if type(fps) not in (int, float) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("sampling.fps must be a positive finite number")
    if any(
        not is_int(value) or value <= 0 for value in (frame_count, width, height)
    ):
        raise ValueError(
            "frames_per_chunk/model_width/model_height must be positive integers"
        )
    if sampling.get("non_overlapping") is not True:
        raise ValueError("sampling.non_overlapping must be true")
    if sampling.get("tail_policy") != "repeat-last-frame-padding":
        raise ValueError("unsupported tail padding policy")
    return float(fps), int(frame_count), int(width), int(height)


def catalog_chunk_record(
    *,
    repo_root: Path,
    output_path: Path,
    status_path: Path,
    sidecar: Mapping[str, Any],
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_path = safe_repo_path(repo_root, str(chunk["prediction_file"]))
    return {
        "id": chunk["id"],
        "index": chunk.get("index"),
        "sample_start_index": chunk.get("sample_start_index"),
        "sample_end_index_exclusive": chunk.get("sample_end_index_exclusive"),
        "start_time_s": float(chunk["start_time_s"]),
        "end_time_s": float(chunk["end_time_s"]),
        "valid_frames": int(chunk["valid_frames"]),
        "pad_frames": int(chunk["pad_frames"]),
        "output": repo_relative(repo_root, output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sidecar["sha256"],
        "source_pt": str(chunk["prediction_file"]),
        "source_pt_bytes": prediction_path.stat().st_size,
        "source_pt_sha256": status["pt_sha256"],
        "source_status": repo_relative(repo_root, status_path),
        "source_rgb": str(chunk["input_dir"]),
        "inference_fingerprint": status["inference_fingerprint"],
        "bounds": manifest["bounds"],
        "sampling": manifest["sampling"],
        "created_at": sidecar.get("created_at"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fully validate every expected 500k OMX4D file, sidecar, and source "
            "status before atomically publishing validation_report.json and catalog.json."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--plan", "--manifest", dest="plan", type=Path, required=True)
    parser.add_argument(
        "--omx4d-root",
        "--output-root",
        dest="omx4d_root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = safe_repo_path(repo_root, args.plan)
    output_root = safe_repo_path(repo_root, args.omx4d_root)
    plan = read_json_object(plan_path)
    fps, frame_count, width, height = scalar_plan_fields(plan)
    videos, plan_failures = plan_summary(plan, frame_count, repo_root)

    errors: list[dict[str, str]] = []
    for failure in plan_failures:
        add_error(errors, "plan", plan_path, failure)
    expected_binaries, expected_sidecars = expected_output_files(output_root, videos)
    expected_payloads = expected_binaries | expected_sidecars
    actual_payloads = actual_payload_files(output_root)
    missing = sorted(expected_payloads - actual_payloads)
    unexpected = sorted(actual_payloads - expected_payloads)
    for path in missing:
        add_error(errors, "file-set", path, "expected payload file is missing")
    for path in unexpected:
        add_error(errors, "file-set", path, "unexpected payload file is present")

    total_chunks = sum(len(video["chunks"]) for video in videos)
    validated_chunks = 0
    hashed_files = 0
    hashed_bytes = 0
    total_output_bytes = 0
    video_catalogs: list[dict[str, Any]] = []
    video_results: dict[str, dict[str, int]] = {}
    ordinal = 0

    for video in videos:
        video_id = str(video["id"])
        chunk_records: list[dict[str, Any]] = []
        video_bytes = 0
        video_validated = 0
        for chunk in video["chunks"]:
            ordinal += 1
            chunk_id = str(chunk["id"])
            scope = f"{video_id}/{chunk_id}"
            output_path = output_root / video_id / f"{chunk_id}.omx4d"
            sidecar_path = output_path.with_suffix(".json")
            status_path = (
                safe_repo_path(repo_root, str(chunk["output_dir"])) / "status.json"
            )
            local_failures: list[str] = []
            manifest: dict[str, Any] | None = None
            sidecar: dict[str, Any] | None = None
            status: dict[str, Any] | None = None
            output_digest: str | None = None
            actual_minimum: np.ndarray | None = None
            actual_maximum: np.ndarray | None = None

            print(
                json.dumps(
                    {
                        "event": "verify",
                        "ordinal": ordinal,
                        "total": total_chunks,
                        "video": video_id,
                        "chunk": chunk_id,
                    }
                ),
                flush=True,
            )

            if output_path.is_symlink():
                local_failures.append("OMX4D payload must not be a symbolic link")
            if sidecar_path.is_symlink():
                local_failures.append("sidecar must not be a symbolic link")
            if output_path.is_symlink():
                pass
            elif not output_path.is_file():
                local_failures.append("OMX4D payload is missing")
            else:
                total_output_bytes += output_path.stat().st_size
                try:
                    manifest, header_length = read_omx4d_manifest(output_path)
                    local_failures.extend(
                        validate_manifest_values(
                            manifest,
                            fps,
                            frame_count,
                            int(chunk["valid_frames"]),
                            width,
                            height,
                        )
                    )
                    descriptors, _, envelope_failures = validate_envelope_contract(
                        output_path, manifest, header_length, frame_count
                    )
                    local_failures.extend(envelope_failures)
                    if len(descriptors) == len(SECTION_ORDER):
                        try:
                            actual_minimum, actual_maximum = scan_positions(
                                output_path, descriptors["positions"]
                            )
                            local_failures.extend(
                                compare_bounds(
                                    "manifest",
                                    manifest.get("bounds"),
                                    actual_minimum,
                                    actual_maximum,
                                )
                            )
                        except (OSError, ValueError) as error:
                            local_failures.append(str(error))
                        try:
                            scan_dynamic_score(
                                output_path, descriptors["dynamicScore"]
                            )
                        except (OSError, ValueError) as error:
                            local_failures.append(str(error))
                        try:
                            local_failures.extend(
                                validate_small_sections(
                                    output_path,
                                    descriptors,
                                    int(chunk["valid_frames"]),
                                )
                            )
                        except (OSError, ValueError) as error:
                            local_failures.append(str(error))
                except (OSError, ValueError) as error:
                    local_failures.append(str(error))

                try:
                    output_digest = sha256(output_path)
                    hashed_files += 1
                    hashed_bytes += output_path.stat().st_size
                except OSError as error:
                    local_failures.append(f"cannot hash OMX4D payload: {error}")

            if sidecar_path.is_symlink():
                pass
            elif not sidecar_path.is_file():
                local_failures.append("OMX4D JSON sidecar is missing")
            else:
                try:
                    sidecar = read_json_object(sidecar_path)
                except ValueError as error:
                    local_failures.append(str(error))
            if status_path.is_symlink():
                local_failures.append("source status.json must not be a symbolic link")
            elif not status_path.is_file():
                local_failures.append("source inference status.json is missing")
            else:
                try:
                    status = read_json_object(status_path)
                except ValueError as error:
                    local_failures.append(str(error))

            if (
                manifest is not None
                and sidecar is not None
                and status is not None
                and output_digest is not None
            ):
                try:
                    local_failures.extend(
                        validate_sidecar_and_status(
                            repo_root=repo_root,
                            output_path=output_path,
                            sidecar=sidecar,
                            status_path=status_path,
                            status=status,
                            manifest=manifest,
                            output_sha256=output_digest,
                            plan=plan,
                            plan_video=video,
                            plan_chunk=chunk,
                            fps=fps,
                            frame_count=frame_count,
                            width=width,
                            height=height,
                        )
                    )
                except (OSError, ValueError) as error:
                    local_failures.append(str(error))
                if actual_minimum is not None and actual_maximum is not None:
                    local_failures.extend(
                        compare_bounds(
                            "sidecar",
                            sidecar.get("bounds"),
                            actual_minimum,
                            actual_maximum,
                        )
                    )

            if local_failures:
                for failure in local_failures:
                    add_error(errors, scope, output_path, failure)
                continue

            assert manifest is not None
            assert sidecar is not None
            assert status is not None
            video_validated += 1
            validated_chunks += 1
            output_bytes = output_path.stat().st_size
            video_bytes += output_bytes
            chunk_records.append(
                catalog_chunk_record(
                    repo_root=repo_root,
                    output_path=output_path,
                    status_path=status_path,
                    sidecar=sidecar,
                    status=status,
                    manifest=manifest,
                    chunk=chunk,
                )
            )

        tail = video["chunks"][-1]
        video_results[video_id] = {
            "expected_files": len(video["chunks"]),
            "validated_files": video_validated,
            "bytes": video_bytes,
            "tail_valid_frames": int(tail["valid_frames"]),
            "tail_pad_frames": int(tail["pad_frames"]),
        }
        video_catalogs.append(
            {
                "id": video_id,
                "source_video": video.get("source"),
                "source_video_sha256": video.get("source_sha256"),
                "source_probe": video.get("source_probe"),
                "crop_filter": video.get("crop_filter"),
                "sampled_frame_count": video.get("sampled_frame_count"),
                "chunk_count": len(video["chunks"]),
                "total_bytes": video_bytes,
                "tail": {
                    "policy": "repeat-last-frame-padding",
                    "chunk": tail["id"],
                    "valid_frames": int(tail["valid_frames"]),
                    "pad_frames": int(tail["pad_frames"]),
                },
                "chunks": chunk_records,
            }
        )

    passed = not errors and validated_chunks == total_chunks
    catalog_path = output_root / "catalog.json"
    catalog_info: dict[str, Any] | None = None
    if passed:
        catalog = {
            "schema": "omnix.omx4d-catalog.v1",
            "created_at": utc_now(),
            "source_plan": repo_relative(repo_root, plan_path),
            "point_count": POINT_COUNT,
            "model_resolution": [width, height],
            "fps": fps,
            "frame_count_per_chunk": frame_count,
            "file_count": validated_chunks,
            "total_bytes": sum(result["bytes"] for result in video_results.values()),
            "sampling": {
                "method": "dynamic-reserved-voxel-v1",
                "dynamic_threshold": DYNAMIC_THRESHOLD,
                "dynamic_reserved_fraction": DYNAMIC_FRACTION,
                "dynamic_points": DYNAMIC_COUNT,
                "spatial_points": SPATIAL_COUNT,
                "stable_point_identities_across_frames": True,
                "source_rgb": True,
                "padded_source_views_excluded": True,
            },
            "cross_chunk_alignment": "none",
            "boundary_policy": "reset-or-crossfade",
            "videos": video_catalogs,
        }
        atomic_json(catalog_path, catalog)
        catalog_info = {
            "path": repo_relative(repo_root, catalog_path),
            "bytes": catalog_path.stat().st_size,
            "sha256": sha256(catalog_path),
        }

    report = {
        "schema": "omnix.omx4d-validation.v1",
        "created_at": utc_now(),
        "result": "passed" if passed else "failed",
        "source_plan": repo_relative(repo_root, plan_path),
        "omx4d_root": repo_relative(repo_root, output_root),
        "full_output_sha256_rehash": True,
        "catalog": catalog_info,
        "scope": {
            "expected_files": total_chunks,
            "expected_sidecars": total_chunks,
            "point_count_per_file": POINT_COUNT,
            "frame_count_per_file": frame_count,
            "fps": fps,
            "model_resolution": [width, height],
            "dynamic_points_per_file": DYNAMIC_COUNT,
            "spatial_points_per_file": SPATIAL_COUNT,
            "videos": {
                video_id: {
                    "files": result["expected_files"],
                    "tail_valid": result["tail_valid_frames"],
                    "tail_pad": result["tail_pad_frames"],
                }
                for video_id, result in video_results.items()
            },
        },
        "results": {
            "validated_files": validated_chunks,
            "hashed_files": hashed_files,
            "hashed_bytes": hashed_bytes,
            "observed_output_bytes": total_output_bytes,
            "missing_files": [str(path) for path in missing],
            "unexpected_files": [str(path) for path in unexpected],
            "per_video": video_results,
        },
        "checks": {
            "file_sets_exact": not missing and not unexpected,
            "binary_magic_schema_descriptors_and_size": passed,
            "exact_500k_80_20_sampling": passed,
            "padded_source_views_excluded": passed,
            "finite_positions_calibration_and_scores": passed,
            "dynamic_score_range": passed,
            "manifest_and_sidecar_bounds_match_positions": passed,
            "sidecars_and_source_status_linked": passed,
            "all_output_sha256_rehashed_and_matched": (
                hashed_files == total_chunks and passed
            ),
            "catalog_written_atomically": catalog_info is not None,
        },
        "errors": errors,
    }
    report_path = output_root / "validation_report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "validation_complete",
                "result": report["result"],
                "validated": validated_chunks,
                "expected": total_chunks,
                "hashed_bytes": hashed_bytes,
                "errors": len(errors),
                "report": str(report_path),
                "catalog": str(catalog_path) if catalog_info else None,
            }
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
