"""Writer and lightweight inspector for the renderer-native OMX4D envelope."""

from __future__ import annotations

import copy
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np
import torch

from .errors import ConversionError, ResourceLimitError


MAGIC = b"OMX4D\r\n\x1a"
SCHEMA_VERSION = 1
PREFIX = struct.Struct("<8sII")
ALIGNMENT = 8
REQUIRED_SECTION_ORDER = (
    "positions",
    "colors",
    "dynamicScore",
    "sourceView",
    "cameraPose",
    "intrinsics",
)
DTYPE_INFO = {
    "float32": (torch.float32, np.dtype("<f4")),
    "uint8": (torch.uint8, np.dtype("u1")),
    "uint16": (torch.uint16, np.dtype("<u2")),
}
SECTION_DTYPES = {
    "positions": "float32",
    "colors": "uint8",
    "dynamicScore": "float32",
    "sourceView": "uint16",
    "cameraPose": "float32",
    "intrinsics": "float32",
}


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _json_bytes(manifest: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            "invalid_manifest", "The OMX4D manifest is not JSON serializable."
        ) from exc


def _section_descriptors(
    sections: Mapping[str, torch.Tensor], offsets: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SECTION_ORDER:
        tensor = sections[name]
        dtype_name = SECTION_DTYPES[name]
        expected_torch_dtype, numpy_dtype = DTYPE_INFO[dtype_name]
        if tensor.dtype != expected_torch_dtype:
            raise ConversionError(
                "invalid_section_dtype",
                f"Section '{name}' must use {dtype_name}.",
            )
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ConversionError(
                "invalid_section_layout",
                f"Section '{name}' must be a contiguous CPU tensor.",
            )
        descriptors[name] = {
            "offset": int(offsets.get(name, 0)),
            "byteLength": int(tensor.numel() * numpy_dtype.itemsize),
            "dtype": dtype_name,
            "shape": [int(dimension) for dimension in tensor.shape],
        }
    return descriptors


def finalize_manifest(
    base_manifest: Mapping[str, Any], sections: Mapping[str, torch.Tensor]
) -> tuple[dict[str, Any], bytes, int]:
    if set(sections) != set(REQUIRED_SECTION_ORDER):
        raise ConversionError(
            "invalid_sections",
            "OMX4D output must contain exactly the six required sections.",
        )

    offsets: dict[str, int] = {}
    for _ in range(16):
        manifest = copy.deepcopy(dict(base_manifest))
        manifest["attributes"] = _section_descriptors(sections, offsets)
        header = _json_bytes(manifest)
        cursor = align(PREFIX.size + len(header))
        next_offsets: dict[str, int] = {}
        for name in REQUIRED_SECTION_ORDER:
            cursor = align(cursor)
            next_offsets[name] = cursor
            cursor += manifest["attributes"][name]["byteLength"]
        if next_offsets == offsets:
            return manifest, header, cursor
        offsets = next_offsets
    raise RuntimeError("OMX4D manifest offsets did not converge")


def _write_padding(file: BinaryIO, target_offset: int) -> None:
    current = file.tell()
    if current > target_offset:
        raise RuntimeError("Attempted to write overlapping OMX4D sections")
    if current < target_offset:
        file.write(b"\0" * (target_offset - current))


def _write_tensor(file: BinaryIO, tensor: torch.Tensor, dtype_name: str) -> None:
    _, output_dtype = DTYPE_INFO[dtype_name]
    flat = tensor.reshape(-1)
    elements_per_chunk = max(1, (8 * 1024 * 1024) // output_dtype.itemsize)
    for start in range(0, flat.numel(), elements_per_chunk):
        array = flat[start : start + elements_per_chunk].numpy()
        # Explicitly normalize byte order even though supported deployment hosts
        # are little-endian.
        if sys.byteorder != "little" and output_dtype.itemsize > 1:
            array = array.byteswap()
        array = array.astype(output_dtype, copy=False)
        file.write(array.tobytes(order="C"))


def write_omx4d(
    output_path: str | os.PathLike[str],
    base_manifest: Mapping[str, Any],
    sections: Mapping[str, torch.Tensor],
    *,
    max_output_bytes: int | None = None,
) -> dict[str, Any]:
    """Write a deterministic OMX4D file and return its finalized manifest."""

    manifest, header, output_size = finalize_manifest(base_manifest, sections)
    if max_output_bytes is not None and output_size > max_output_bytes:
        raise ResourceLimitError(
            "output_byte_limit_exceeded",
            "The sampled renderer payload would exceed the configured output limit.",
            details={"received": output_size, "limit": max_output_bytes},
        )

    path = Path(output_path)
    with path.open("wb") as file:
        file.write(PREFIX.pack(MAGIC, SCHEMA_VERSION, len(header)))
        file.write(header)
        for name in REQUIRED_SECTION_ORDER:
            descriptor = manifest["attributes"][name]
            _write_padding(file, descriptor["offset"])
            _write_tensor(file, sections[name], descriptor["dtype"])
        file.flush()
        os.fsync(file.fileno())
    if path.stat().st_size != output_size:
        raise RuntimeError("Written OMX4D size does not match its manifest")
    return manifest


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read just the prefix and manifest, primarily for tests and diagnostics."""

    with Path(path).open("rb") as file:
        prefix = file.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ConversionError("invalid_omx4d", "OMX4D prefix is truncated.")
        magic, version, header_length = PREFIX.unpack(prefix)
        if magic != MAGIC or version != SCHEMA_VERSION:
            raise ConversionError("invalid_omx4d", "OMX4D magic or version is invalid.")
        try:
            manifest = json.loads(file.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversionError("invalid_omx4d", "OMX4D manifest is invalid.") from exc
    if not isinstance(manifest, dict):
        raise ConversionError("invalid_omx4d", "OMX4D manifest must be an object.")
    return manifest
