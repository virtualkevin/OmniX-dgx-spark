"""Safe `.pt` to `.omx4d` conversion pipeline."""

from __future__ import annotations

import math
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from .config import IngestionLimits, MIB
from .errors import ConversionError, ResourceLimitError
from .omx4d import write_omx4d
from .sampling import sample_predictions
from .schema import validate_predictions


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    point_budget: int = 100_000
    fps: float = 15.0
    name: str = "OmniX predictions"
    dynamic_threshold: float = 0.5
    dynamic_reserved_fraction: float = 0.25


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConversionControl:
    deadline: float | None = None
    cancelled: threading.Event | None = None

    def checkpoint(self) -> None:
        if self.cancelled is not None and self.cancelled.is_set():
            raise ConversionError(
                "conversion_cancelled", "Conversion was cancelled.", status_code=408
            )
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ConversionError(
                "conversion_timeout",
                "Conversion exceeded the configured wall-clock limit.",
                status_code=408,
            )


def sanitize_dataset_name(value: str | None) -> str:
    if not value:
        return "OmniX predictions"
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._")
    return (cleaned or "OmniX predictions")[:80]


def validate_options(
    options: ConversionOptions, limits: IngestionLimits
) -> ConversionOptions:
    if options.point_budget <= 0 or options.point_budget > limits.max_point_budget:
        raise ConversionError(
            "invalid_point_budget",
            f"Point budget must be between 1 and {limits.max_point_budget}.",
            status_code=400,
        )
    if not math.isfinite(options.fps) or not 0.1 <= options.fps <= 240.0:
        raise ConversionError(
            "invalid_fps", "FPS must be a finite value between 0.1 and 240.", status_code=400
        )
    if not 0.0 <= options.dynamic_threshold <= 1.0:
        raise ConversionError(
            "invalid_dynamic_threshold",
            "Dynamic threshold must be in [0, 1].",
            status_code=400,
        )
    if not 0.0 <= options.dynamic_reserved_fraction <= 1.0:
        raise ConversionError(
            "invalid_dynamic_fraction",
            "Dynamic reserve fraction must be in [0, 1].",
            status_code=400,
        )
    return ConversionOptions(
        point_budget=int(options.point_budget),
        fps=float(options.fps),
        name=sanitize_dataset_name(options.name),
        dynamic_threshold=float(options.dynamic_threshold),
        dynamic_reserved_fraction=float(options.dynamic_reserved_fraction),
    )


def inspect_pt_archive(
    input_path: str | os.PathLike[str], limits: IngestionLimits
) -> None:
    """Reject malformed/oversized ZIP containers before invoking PyTorch."""

    path = Path(input_path)
    try:
        upload_size = path.stat().st_size
    except OSError as exc:
        raise ConversionError("unreadable_upload", "The uploaded file is unavailable.") from exc
    if upload_size <= 0:
        raise ConversionError("empty_upload", "The uploaded file is empty.")
    if upload_size > limits.max_upload_bytes:
        raise ResourceLimitError(
            "upload_too_large",
            "The uploaded file exceeds the configured size limit.",
            details={"received": upload_size, "limit": limits.max_upload_bytes},
        )
    if not zipfile.is_zipfile(path):
        raise ConversionError(
            "invalid_pt_archive",
            "The file is not a supported ZIP-based torch.save archive.",
        )

    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.infolist()
            if not entries or len(entries) > limits.max_zip_entries:
                raise ResourceLimitError(
                    "archive_entry_limit_exceeded",
                    "The .pt archive contains too many entries.",
                    details={"received": len(entries), "limit": limits.max_zip_entries},
                )
            names: set[str] = set()
            total_uncompressed = 0
            has_pickle = False
            has_version = False
            for entry in entries:
                name = entry.filename
                parts = PurePosixPath(name).parts
                if (
                    not name
                    or "\\" in name
                    or PurePosixPath(name).is_absolute()
                    or ".." in parts
                    or name in names
                ):
                    raise ConversionError(
                        "invalid_pt_archive", "The .pt archive has an unsafe entry name."
                    )
                names.add(name)
                if entry.flag_bits & 0x1:
                    raise ConversionError(
                        "invalid_pt_archive", "Encrypted .pt archives are not supported."
                    )
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ConversionError(
                        "invalid_pt_archive",
                        "The .pt archive uses an unsupported compression method.",
                    )
                total_uncompressed += entry.file_size
                if total_uncompressed > limits.max_archive_uncompressed_bytes:
                    raise ResourceLimitError(
                        "archive_byte_limit_exceeded",
                        "The expanded .pt archive exceeds the configured byte limit.",
                        details={
                            "received": total_uncompressed,
                            "limit": limits.max_archive_uncompressed_bytes,
                        },
                    )
                if entry.file_size >= MIB:
                    ratio = entry.file_size / max(1, entry.compress_size)
                    if ratio > limits.max_compression_ratio:
                        raise ResourceLimitError(
                            "archive_compression_ratio_exceeded",
                            "The .pt archive has a suspicious compression ratio.",
                        )
                has_pickle |= name.endswith("/data.pkl") or name == "data.pkl"
                has_version |= name.endswith("/version") or name == "version"
            if not has_pickle or not has_version:
                raise ConversionError(
                    "invalid_pt_archive",
                    "The ZIP file is not a supported torch.save archive.",
                )
    except zipfile.BadZipFile as exc:
        raise ConversionError(
            "invalid_pt_archive", "The .pt ZIP container is malformed."
        ) from exc


def safe_torch_load(input_path: str | os.PathLike[str]) -> Any:
    """Load through PyTorch's restricted weights-only unpickler, with no fallback."""

    try:
        return torch.load(
            input_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        # Intentionally do not surface the loader's exception text: pickle
        # errors can include attacker-controlled class/module names and paths.
        raise ConversionError(
            "unsafe_or_malformed_pt",
            "PyTorch could not safely load this file as a weights-only tensor archive.",
        ) from exc


def convert_pt_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    options: ConversionOptions | None = None,
    limits: IngestionLimits | None = None,
    control: ConversionControl | None = None,
    source_rgb: torch.Tensor | None = None,
) -> ConversionResult:
    """Convert one validated OmniX `.pt` archive into a browser payload."""

    limits = limits or IngestionLimits()
    options = validate_options(options or ConversionOptions(), limits)
    control = control or ConversionControl()
    control.checkpoint()
    inspect_pt_archive(input_path, limits)
    control.checkpoint()
    raw = safe_torch_load(input_path)
    control.checkpoint()
    predictions = validate_predictions(raw, limits)
    control.checkpoint()
    sampled = sample_predictions(
        predictions,
        options.point_budget,
        dynamic_threshold=options.dynamic_threshold,
        dynamic_reserved_fraction=options.dynamic_reserved_fraction,
        check_finite=False,
        source_rgb=source_rgb,
    )
    control.checkpoint()

    frame_count = predictions.frame_count
    warnings = [
        "The .pt format has no timing metadata; playback uses the selected FPS.",
        "World-space units are not recorded in the OmniX output.",
    ]
    if source_rgb is None:
        warnings.insert(0, "Source RGB was not supplied; colors use a stable source/dynamic palette.")
    base_manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "name": options.name,
        "fps": options.fps,
        "frameCount": frame_count,
        "durationSeconds": frame_count / options.fps,
        "sourceViewCount": predictions.source_view_count,
        "pointCount": sampled.point_count,
        "coordinateSystem": "threejs-right-handed-y-up",
        "units": "unknown",
        "primitive": "points",
        "bounds": sampled.bounds,
        "sampling": sampled.sampling,
        "warnings": warnings,
    }
    sections = {
        "positions": sampled.positions,
        "colors": sampled.colors,
        "dynamicScore": sampled.dynamic_score,
        "sourceView": sampled.source_view,
        "cameraPose": sampled.camera_pose,
        "intrinsics": sampled.intrinsics,
    }
    manifest = write_omx4d(
        output_path,
        base_manifest,
        sections,
        max_output_bytes=limits.max_output_bytes,
    )
    control.checkpoint()
    return ConversionResult(output_path=Path(output_path), manifest=manifest)
