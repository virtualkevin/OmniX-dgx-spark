"""Configuration and resource ceilings for untrusted `.pt` ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass


MIB = 1024 * 1024
GIB = 1024 * MIB


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    """Hard limits checked before derived trajectory copies are materialized."""

    max_upload_bytes: int = GIB
    max_archive_uncompressed_bytes: int = 8 * GIB
    max_total_tensor_bytes: int = 8 * GIB
    max_output_bytes: int = 2 * GIB
    max_source_views: int = 64
    max_frames: int = 600
    max_source_pixels: int = 16_000_000
    max_point_budget: int = 200_000
    max_zip_entries: int = 512
    max_compression_ratio: float = 200.0
    finite_check_chunk_elements: int = 8_000_000

    @classmethod
    def from_env(cls) -> "IngestionLimits":
        return cls(
            max_upload_bytes=_env_int("OMX_MAX_UPLOAD_BYTES", GIB),
            max_archive_uncompressed_bytes=_env_int(
                "OMX_MAX_ARCHIVE_BYTES", 8 * GIB
            ),
            max_total_tensor_bytes=_env_int("OMX_MAX_TENSOR_BYTES", 8 * GIB),
            max_output_bytes=_env_int("OMX_MAX_OUTPUT_BYTES", 2 * GIB),
            max_source_views=_env_int("OMX_MAX_SOURCE_VIEWS", 64),
            max_frames=_env_int("OMX_MAX_FRAMES", 600),
            max_source_pixels=_env_int("OMX_MAX_SOURCE_PIXELS", 16_000_000),
            max_point_budget=_env_int("OMX_MAX_POINT_BUDGET", 200_000),
            max_zip_entries=_env_int("OMX_MAX_ZIP_ENTRIES", 512),
            max_compression_ratio=_env_float("OMX_MAX_COMPRESSION_RATIO", 200.0),
            finite_check_chunk_elements=_env_int(
                "OMX_FINITE_CHUNK_ELEMENTS", 8_000_000
            ),
        )


@dataclass(frozen=True, slots=True)
class ServerSettings:
    limits: IngestionLimits = IngestionLimits()
    conversion_timeout_seconds: float = 300.0
    upload_chunk_bytes: int = MIB
    temp_directory: str | None = None

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            limits=IngestionLimits.from_env(),
            conversion_timeout_seconds=_env_float("OMX_CONVERSION_TIMEOUT", 300.0),
            upload_chunk_bytes=_env_int("OMX_UPLOAD_CHUNK_BYTES", MIB),
            temp_directory=os.getenv("OMX_TEMP_DIRECTORY") or None,
        )
