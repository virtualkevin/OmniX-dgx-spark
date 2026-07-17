from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from viewer.server.omx4d import MAGIC, PREFIX, read_manifest, write_omx4d
from viewer.server.sampling import sample_predictions
from viewer.server.schema import validate_predictions


def _sections(sampled):
    return {
        "positions": sampled.positions,
        "colors": sampled.colors,
        "dynamicScore": sampled.dynamic_score,
        "sourceView": sampled.source_view,
        "cameraPose": sampled.camera_pose,
        "intrinsics": sampled.intrinsics,
    }


def _base_manifest(sampled) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "name": "tiny",
        "fps": 15.0,
        "frameCount": 3,
        "durationSeconds": 0.2,
        "sourceViewCount": 2,
        "pointCount": sampled.point_count,
        "coordinateSystem": "threejs-right-handed-y-up",
        "units": "unknown",
        "primitive": "points",
        "bounds": sampled.bounds,
        "sampling": sampled.sampling,
        "warnings": [],
    }


def test_sampling_is_deterministic_and_preserves_identity(
    valid_predictions: dict[str, torch.Tensor],
) -> None:
    predictions = validate_predictions(valid_predictions)
    first = sample_predictions(predictions, 9)
    second = sample_predictions(predictions, 9)
    assert first.sampling["identityHash"] == second.sampling["identityHash"]
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.source_view, second.source_view)
    assert first.positions.shape == (3, 9, 3)
    # Every fixture identity moves +0.25 in x at each target frame.
    assert torch.allclose(
        first.positions[1, :, 0] - first.positions[0, :, 0],
        torch.full((9,), 0.25),
    )


def test_sampling_converts_positions_and_cameras_to_three_coordinates(
    valid_predictions: dict[str, torch.Tensor],
) -> None:
    sampled = sample_predictions(validate_predictions(valid_predictions), 24)
    original = valid_predictions["trajectory"].permute(1, 0, 2, 3, 4).reshape(3, 24, 3)
    expected = original.clone()
    expected[..., 1:] *= -1
    assert torch.allclose(sampled.positions, expected)
    assert torch.allclose(
        sampled.camera_pose[0, :3, 3], torch.tensor([1.0, -2.0, -3.0])
    )
    assert sampled.bounds["min"] == pytest.approx(
        expected.reshape(-1, 3).amin(0).tolist()
    )


def test_binary_manifest_offsets_and_lengths_are_integral(
    tmp_path: Path, valid_predictions: dict[str, torch.Tensor]
) -> None:
    sampled = sample_predictions(validate_predictions(valid_predictions), 9)
    output = tmp_path / "tiny.omx4d"
    manifest = write_omx4d(output, _base_manifest(sampled), _sections(sampled))
    assert read_manifest(output) == manifest

    raw = output.read_bytes()
    magic, version, header_length = PREFIX.unpack(raw[: PREFIX.size])
    assert magic == MAGIC
    assert version == 1
    assert json.loads(raw[PREFIX.size : PREFIX.size + header_length]) == manifest
    for descriptor in manifest["attributes"].values():
        assert descriptor["offset"] % 8 == 0
        assert descriptor["offset"] + descriptor["byteLength"] <= len(raw)
    positions = manifest["attributes"]["positions"]
    decoded = np.frombuffer(
        raw,
        dtype="<f4",
        count=positions["byteLength"] // 4,
        offset=positions["offset"],
    ).reshape(positions["shape"])
    np.testing.assert_array_equal(decoded, sampled.positions.numpy())
