"""Deterministic, stable-identity sampling for dense OmniX trajectories."""

from __future__ import annotations

import colorsys
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from .errors import ConversionError
from .schema import ValidatedPredictions


SAMPLING_METHOD = "dynamic-reserved-voxel-v1"


@dataclass(frozen=True, slots=True)
class SampledPointCloud:
    positions: torch.Tensor
    colors: torch.Tensor
    dynamic_score: torch.Tensor
    source_view: torch.Tensor
    camera_pose: torch.Tensor
    intrinsics: torch.Tensor
    bounds: dict[str, list[float]]
    sampling: dict[str, object]

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[1])


def _evenly_spaced(values: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return values[:0]
    if count >= values.numel():
        return values
    # Midpoints of equal-width index intervals. For count <= size this cannot
    # produce duplicate positions and does not depend on an RNG implementation.
    positions = torch.floor(
        (torch.arange(count, dtype=torch.float64) + 0.5)
        * (values.numel() / count)
    ).to(torch.int64)
    return values[positions]


def _valid_identity_mask(trajectory: torch.Tensor) -> torch.Tensor:
    source_views, frames, height, width, _ = trajectory.shape
    valid = torch.ones(
        (source_views, height, width), dtype=torch.bool, device="cpu"
    )
    # Iterating over time keeps the largest temporary at S*H*W instead of
    # allocating a boolean tensor as large as the full trajectory.
    for frame in range(frames):
        valid &= torch.isfinite(trajectory[:, frame]).all(dim=-1)
    return valid.reshape(-1)


def _reference_bounds(
    predictions: ValidatedPredictions, identity_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = predictions.height, predictions.width
    pixels_per_view = height * width
    minimum = torch.full((3,), torch.inf, dtype=torch.float32)
    maximum = torch.full((3,), -torch.inf, dtype=torch.float32)
    for source in range(predictions.source_view_count):
        view_mask = identity_mask[
            source * pixels_per_view : (source + 1) * pixels_per_view
        ]
        if not bool(view_mask.any().item()):
            continue
        points = predictions.trajectory[source, 0].reshape(-1, 3)[view_mask]
        minimum = torch.minimum(minimum, points.amin(dim=0))
        maximum = torch.maximum(maximum, points.amax(dim=0))
    return minimum, maximum


def _voxel_sample(
    predictions: ValidatedPredictions,
    candidate_mask: torch.Tensor,
    count: int,
) -> torch.Tensor:
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if count >= candidates.numel():
        return candidates
    if count <= 0:
        return candidates[:0]

    minimum, maximum = _reference_bounds(predictions, candidate_mask)
    extent = maximum - minimum
    active_axes = extent > 1e-12
    active_count = int(active_axes.sum().item())
    if active_count == 0:
        return _evenly_spaced(candidates, count)

    # Use twice as many normalized cells as desired samples, then select one
    # stable representative per occupied cell. Normalizing each active axis
    # prevents a long scene dimension from starving thin dimensions.
    cells_per_axis = max(1, math.ceil((count * 2) ** (1.0 / active_count)))
    dimensions = torch.where(
        active_axes,
        torch.full((3,), cells_per_axis, dtype=torch.int64),
        torch.ones((3,), dtype=torch.int64),
    )

    pixels_per_view = predictions.height * predictions.width
    source = torch.div(candidates, pixels_per_view, rounding_mode="floor")
    pixel = candidates.remainder(pixels_per_view)
    y = torch.div(pixel, predictions.width, rounding_mode="floor")
    x = pixel.remainder(predictions.width)
    points = predictions.trajectory[source, 0, y, x, :]
    safe_extent = torch.where(active_axes, extent, torch.ones_like(extent))
    normalized = ((points - minimum) / safe_extent).clamp(0.0, 1.0)
    cell = torch.floor(normalized * dimensions.to(torch.float32)).to(torch.int64)
    cell = torch.minimum(cell, dimensions - 1)
    key = (cell[:, 0] * dimensions[1] + cell[:, 1]) * dimensions[2] + cell[:, 2]

    order = torch.argsort(key, stable=True)
    sorted_keys = key[order]
    first_in_cell = torch.ones(sorted_keys.numel(), dtype=torch.bool)
    first_in_cell[1:] = sorted_keys[1:] != sorted_keys[:-1]
    representatives = candidates[order[first_in_cell]]

    if representatives.numel() >= count:
        return _evenly_spaced(representatives, count)

    # Sparse/degenerate clouds can occupy fewer cells than requested. Fill the
    # remainder from non-representative identities at a deterministic stride.
    chosen_mask = torch.zeros_like(candidate_mask)
    chosen_mask[representatives] = True
    fill_candidates = candidates[~chosen_mask[candidates]]
    fill = _evenly_spaced(fill_candidates, count - representatives.numel())
    return torch.cat((representatives, fill))


def _identity_hash(indices: torch.Tensor) -> str:
    array = indices.contiguous().numpy().astype(np.dtype("<i8"), copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _source_palette(source_view_count: int) -> torch.Tensor:
    golden_ratio = 0.6180339887498949
    colors = []
    for source in range(source_view_count):
        red, green, blue = colorsys.hsv_to_rgb(
            (0.08 + source * golden_ratio) % 1.0, 0.62, 0.95
        )
        colors.append([round(red * 255), round(green * 255), round(blue * 255)])
    return torch.tensor(colors, dtype=torch.float32)


def sample_predictions(
    predictions: ValidatedPredictions,
    point_budget: int,
    *,
    dynamic_threshold: float = 0.5,
    dynamic_reserved_fraction: float = 0.25,
    check_finite: bool = True,
    source_rgb: torch.Tensor | None = None,
) -> SampledPointCloud:
    """Sample points once, preserving the same identities in every frame."""

    if point_budget <= 0:
        raise ConversionError("invalid_point_budget", "Point budget must be positive.")
    if not 0.0 <= dynamic_threshold <= 1.0:
        raise ConversionError(
            "invalid_dynamic_threshold", "Dynamic threshold must be in [0, 1]."
        )
    if not 0.0 <= dynamic_reserved_fraction <= 1.0:
        raise ConversionError(
            "invalid_dynamic_fraction", "Dynamic reserve fraction must be in [0, 1]."
        )

    total_identities = (
        predictions.source_view_count * predictions.height * predictions.width
    )
    valid = (
        _valid_identity_mask(predictions.trajectory)
        if check_finite
        else torch.ones(total_identities, dtype=torch.bool)
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ConversionError(
            "no_valid_points",
            "No point stays finite across every trajectory frame.",
        )

    selected_count = min(point_budget, valid_count)
    flat_score = predictions.dynamic_score.reshape(-1)
    reserve_target = min(
        selected_count, int(round(selected_count * dynamic_reserved_fraction))
    )
    dynamic_candidates = torch.nonzero(
        valid & (flat_score >= dynamic_threshold), as_tuple=False
    ).flatten()
    dynamic_take = min(reserve_target, int(dynamic_candidates.numel()))
    if dynamic_take:
        ranking = torch.argsort(
            flat_score[dynamic_candidates], descending=True, stable=True
        )
        dynamic_selected = dynamic_candidates[ranking[:dynamic_take]]
    else:
        dynamic_selected = dynamic_candidates[:0]

    spatial_mask = valid.clone()
    spatial_mask[dynamic_selected] = False
    spatial_selected = _voxel_sample(
        predictions, spatial_mask, selected_count - dynamic_selected.numel()
    )
    identities = torch.cat((dynamic_selected, spatial_selected)).sort().values
    if identities.numel() != selected_count:
        raise RuntimeError("Sampler did not produce the requested unique point count")

    pixels_per_view = predictions.height * predictions.width
    source = torch.div(identities, pixels_per_view, rounding_mode="floor")
    pixel = identities.remainder(pixels_per_view)
    y = torch.div(pixel, predictions.width, rounding_mode="floor")
    x = pixel.remainder(predictions.width)

    # Advanced indexing selects only K*T*3 values from the dense source tensor;
    # no T*S*H*W transpose/copy is created.
    by_identity = predictions.trajectory[source, :, y, x, :]
    if tuple(by_identity.shape) != (selected_count, predictions.frame_count, 3):
        raise RuntimeError("Unexpected trajectory gather shape")
    positions = by_identity.permute(1, 0, 2).contiguous()
    # Preserve handedness while converting OpenCV +y-down/+z-forward world
    # coordinates to Three.js' +y-up/-z-forward basis.
    positions[..., 1:].mul_(-1)
    selected_score = flat_score[identities].contiguous()

    if source_rgb is not None:
        expected_rgb_shape = (
            predictions.source_view_count,
            predictions.height,
            predictions.width,
            3,
        )
        if source_rgb.dtype != torch.uint8 or tuple(source_rgb.shape) != expected_rgb_shape:
            raise ConversionError(
                "invalid_source_rgb",
                "Companion RGB must be uint8 [source view, height, width, 3] data.",
            )
        colors = source_rgb.contiguous().reshape(-1, 3)[identities].contiguous()
    else:
        palette = _source_palette(predictions.source_view_count)
        base_color = palette[source]
        accent = torch.tensor([255.0, 72.0, 86.0], dtype=torch.float32)
        blend = selected_score.clamp(0.0, 1.0).unsqueeze(1) * 0.45
        colors = torch.round(base_color * (1.0 - blend) + accent * blend).to(torch.uint8)

    camera_pose = torch.zeros(
        (predictions.source_view_count, 4, 4), dtype=torch.float32
    )
    camera_pose[:, :3, :] = predictions.camera_pose
    camera_pose[:, 3, 3] = 1.0
    basis = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    camera_pose = (basis @ camera_pose @ basis).contiguous()

    minimum = positions.reshape(-1, 3).amin(dim=0)
    maximum = positions.reshape(-1, 3).amax(dim=0)
    bounds = {
        "min": [float(value) for value in minimum.tolist()],
        "max": [float(value) for value in maximum.tolist()],
    }
    sampling: dict[str, object] = {
        "method": SAMPLING_METHOD,
        "requestedPointCount": point_budget,
        "selectedPointCount": selected_count,
        "validCandidateCount": valid_count,
        "dynamicThreshold": dynamic_threshold,
        "dynamicReservedFraction": dynamic_reserved_fraction,
        "dynamicSelectedPointCount": dynamic_take,
        "identityHash": _identity_hash(identities),
    }

    return SampledPointCloud(
        positions=positions,
        colors=colors.contiguous(),
        dynamic_score=selected_score,
        source_view=source.to(torch.uint16).contiguous(),
        camera_pose=camera_pose,
        intrinsics=predictions.intrinsics.contiguous(),
        bounds=bounds,
        sampling=sampling,
    )
