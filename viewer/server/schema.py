"""Strict validation for the only supported OmniX tensor schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .config import IngestionLimits
from .errors import ConversionError, ResourceLimitError


REQUIRED_KEYS = frozenset(
    {"trajectory", "camera_pose", "intrinsics", "pts3d_dynamic_score"}
)


@dataclass(frozen=True, slots=True)
class ValidatedPredictions:
    trajectory: torch.Tensor
    camera_pose: torch.Tensor
    intrinsics: torch.Tensor
    dynamic_score: torch.Tensor

    @property
    def source_view_count(self) -> int:
        return int(self.trajectory.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.trajectory.shape[1])

    @property
    def height(self) -> int:
        return int(self.trajectory.shape[2])

    @property
    def width(self) -> int:
        return int(self.trajectory.shape[3])


def _safe_key(key: Any) -> str:
    if isinstance(key, str):
        return key[:80]
    return f"<{type(key).__name__}>"


def _validate_tensor_base(name: str, value: Any, rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ConversionError(
            "invalid_tensor_type",
            f"'{name}' must be a PyTorch tensor, not {type(value).__name__}.",
        )
    if value.layout != torch.strided or value.is_sparse or value.is_quantized:
        raise ConversionError(
            "unsupported_tensor_layout",
            f"'{name}' must be a dense, non-quantized strided tensor.",
        )
    if value.dtype != torch.float32:
        raise ConversionError(
            "unsupported_tensor_dtype",
            f"'{name}' must use float32 values.",
            details={"field": name, "received": str(value.dtype)},
        )
    if value.device.type != "cpu":
        raise ConversionError(
            "unsupported_tensor_device",
            f"'{name}' could not be normalized to CPU memory.",
        )
    if value.ndim != rank:
        raise ConversionError(
            "invalid_tensor_rank",
            f"'{name}' must have rank {rank}.",
            details={"field": name, "shape": list(value.shape)},
        )
    if any(int(dimension) <= 0 for dimension in value.shape):
        raise ConversionError(
            "invalid_tensor_shape",
            f"'{name}' must have only positive dimensions.",
            details={"field": name, "shape": list(value.shape)},
        )
    return value.detach()


def _all_finite(tensor: torch.Tensor, chunk_elements: int) -> bool:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        if not bool(torch.isfinite(flat[start : start + chunk_elements]).all().item()):
            return False
    return True


def _rotation_determinants(rotation: torch.Tensor) -> torch.Tensor:
    a, b, c = rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2]
    d, e, f = rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2]
    g, h, i = rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def validate_predictions(
    raw: Any, limits: IngestionLimits | None = None
) -> ValidatedPredictions:
    """Validate and normalize the exact plain-tensor OmniX output contract."""

    limits = limits or IngestionLimits()
    if type(raw) is not dict:
        raise ConversionError(
            "invalid_root_object",
            "The .pt root must be a plain dictionary containing four tensors.",
        )

    actual_keys = set(raw.keys())
    if actual_keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - actual_keys, key=str)
        unexpected = sorted(
            (_safe_key(key) for key in actual_keys - REQUIRED_KEYS), key=str
        )[:16]
        raise ConversionError(
            "invalid_schema_keys",
            "The .pt keys do not match the supported OmniX output schema.",
            details={"missing": missing, "unexpected": unexpected},
        )

    trajectory = _validate_tensor_base("trajectory", raw["trajectory"], 5)
    camera_pose = _validate_tensor_base("camera_pose", raw["camera_pose"], 3)
    intrinsics = _validate_tensor_base("intrinsics", raw["intrinsics"], 3)
    dynamic_score = _validate_tensor_base(
        "pts3d_dynamic_score", raw["pts3d_dynamic_score"], 3
    )

    source_views, frames, height, width, xyz = map(int, trajectory.shape)
    if xyz != 3:
        raise ConversionError(
            "invalid_trajectory_shape",
            "'trajectory' must end in an xyz dimension of length 3.",
            details={"shape": list(trajectory.shape)},
        )
    expected_shapes = {
        "camera_pose": [source_views, 3, 4],
        "intrinsics": [source_views, 3, 3],
        "pts3d_dynamic_score": [source_views, height, width],
    }
    actual_shapes = {
        "camera_pose": list(camera_pose.shape),
        "intrinsics": list(intrinsics.shape),
        "pts3d_dynamic_score": list(dynamic_score.shape),
    }
    mismatches = {
        name: {"expected": expected_shapes[name], "received": actual_shapes[name]}
        for name in expected_shapes
        if expected_shapes[name] != actual_shapes[name]
    }
    if mismatches:
        raise ConversionError(
            "incompatible_tensor_shapes",
            "Camera and score shapes must match the trajectory source/image dimensions.",
            details={"mismatches": mismatches},
        )

    source_pixels = source_views * height * width
    if source_views > limits.max_source_views:
        raise ResourceLimitError(
            "source_view_limit_exceeded",
            f"The dataset has {source_views} source views; the limit is {limits.max_source_views}.",
        )
    if frames > limits.max_frames:
        raise ResourceLimitError(
            "frame_limit_exceeded",
            f"The dataset has {frames} frames; the limit is {limits.max_frames}.",
        )
    if source_pixels > limits.max_source_pixels:
        raise ResourceLimitError(
            "source_pixel_limit_exceeded",
            "The source-view pixel count exceeds the configured conversion limit.",
            details={"received": source_pixels, "limit": limits.max_source_pixels},
        )

    tensors = (trajectory, camera_pose, intrinsics, dynamic_score)
    logical_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if logical_bytes > limits.max_total_tensor_bytes:
        raise ResourceLimitError(
            "tensor_byte_limit_exceeded",
            "The decoded tensor data exceeds the configured byte limit.",
            details={"received": logical_bytes, "limit": limits.max_total_tensor_bytes},
        )

    # Only normalize layout after all shape and byte ceilings have been checked.
    trajectory, camera_pose, intrinsics, dynamic_score = (
        tensor.contiguous() for tensor in tensors
    )
    for name, tensor in (
        ("trajectory", trajectory),
        ("camera_pose", camera_pose),
        ("intrinsics", intrinsics),
        ("pts3d_dynamic_score", dynamic_score),
    ):
        if not _all_finite(tensor, limits.finite_check_chunk_elements):
            raise ConversionError(
                "non_finite_tensor",
                f"'{name}' contains NaN or infinite values.",
                details={"field": name},
            )

    score_min = float(dynamic_score.min().item())
    score_max = float(dynamic_score.max().item())
    if score_min < -1e-6 or score_max > 1.0 + 1e-6:
        raise ConversionError(
            "invalid_dynamic_score_range",
            "'pts3d_dynamic_score' must contain probabilities in [0, 1].",
            details={"minimum": score_min, "maximum": score_max},
        )

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    expected_bottom = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if bool((fx <= 0).any().item()) or bool((fy <= 0).any().item()):
        raise ConversionError(
            "invalid_intrinsics",
            "Camera focal lengths must be positive.",
        )
    if not bool(
        torch.isclose(intrinsics[:, 2, :], expected_bottom, atol=1e-4, rtol=1e-4)
        .all()
        .item()
    ):
        raise ConversionError(
            "invalid_intrinsics",
            "Each intrinsic matrix must have homogeneous row [0, 0, 1].",
        )

    rotation = camera_pose[:, :, :3]
    determinants = _rotation_determinants(rotation)
    gram = rotation.transpose(1, 2) @ rotation
    identity = torch.eye(3, dtype=torch.float32).expand_as(gram)
    if bool((determinants <= 0.5).any().item()) or bool(
        (determinants >= 1.5).any().item()
    ) or not bool(torch.isclose(gram, identity, atol=5e-2, rtol=5e-2).all().item()):
        raise ConversionError(
            "invalid_camera_pose",
            "Camera pose rotations must be non-singular, right-handed, and approximately orthonormal.",
        )

    return ValidatedPredictions(
        trajectory=trajectory,
        camera_pose=camera_pose,
        intrinsics=intrinsics,
        dynamic_score=dynamic_score,
    )
