from __future__ import annotations

import pytest
import torch

from viewer.server.config import IngestionLimits
from viewer.server.errors import ConversionError, ResourceLimitError
from viewer.server.schema import validate_predictions


def test_valid_schema_is_normalized(valid_predictions: dict[str, torch.Tensor]) -> None:
    valid_predictions["trajectory"] = valid_predictions["trajectory"].transpose(2, 3)
    # Keep related image shapes consistent after making the trajectory non-contiguous.
    valid_predictions["pts3d_dynamic_score"] = valid_predictions[
        "pts3d_dynamic_score"
    ].transpose(1, 2)
    validated = validate_predictions(valid_predictions)
    assert validated.trajectory.is_contiguous()
    assert validated.dynamic_score.is_contiguous()
    assert validated.frame_count == 3


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda data: data.update({"unexpected": torch.tensor(1.0)}), "invalid_schema_keys"),
        (lambda data: data.pop("intrinsics"), "invalid_schema_keys"),
        (
            lambda data: data.update({"trajectory": data["trajectory"].double()}),
            "unsupported_tensor_dtype",
        ),
        (
            lambda data: data.update(
                {"camera_pose": torch.zeros((2, 4, 4), dtype=torch.float32)}
            ),
            "incompatible_tensor_shapes",
        ),
    ],
)
def test_schema_rejects_wrong_contract(
    valid_predictions: dict[str, torch.Tensor], mutation, error_code: str
) -> None:
    mutation(valid_predictions)
    with pytest.raises(ConversionError) as caught:
        validate_predictions(valid_predictions)
    assert caught.value.code == error_code


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_schema_rejects_non_finite_values(
    valid_predictions: dict[str, torch.Tensor], value: float
) -> None:
    valid_predictions["trajectory"][0, 0, 0, 0, 0] = value
    with pytest.raises(ConversionError) as caught:
        validate_predictions(valid_predictions)
    assert caught.value.code == "non_finite_tensor"


def test_schema_rejects_resource_limit(
    valid_predictions: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(ResourceLimitError) as caught:
        validate_predictions(
            valid_predictions, IngestionLimits(max_source_pixels=1)
        )
    assert caught.value.status_code == 413
    assert caught.value.code == "source_pixel_limit_exceeded"


def test_schema_validates_camera_matrices(
    valid_predictions: dict[str, torch.Tensor],
) -> None:
    valid_predictions["intrinsics"][0, 0, 0] = 0
    with pytest.raises(ConversionError) as caught:
        validate_predictions(valid_predictions)
    assert caught.value.code == "invalid_intrinsics"
