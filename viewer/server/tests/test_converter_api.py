from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from viewer.server.app import create_app
from viewer.server.config import IngestionLimits, ServerSettings
from viewer.server.converter import ConversionOptions, convert_pt_file
from viewer.server.errors import ConversionError
from viewer.server.omx4d import MAGIC, read_manifest


def test_end_to_end_conversion(
    tmp_path: Path, valid_predictions: dict[str, torch.Tensor]
) -> None:
    source = tmp_path / "predictions.pt"
    output = tmp_path / "result.omx4d"
    torch.save(valid_predictions, source)
    result = convert_pt_file(
        source,
        output,
        options=ConversionOptions(point_budget=7, fps=12.0, name="tiny"),
    )
    assert result.manifest == read_manifest(output)
    assert result.manifest["pointCount"] == 7
    assert result.manifest["durationSeconds"] == pytest.approx(0.25)
    assert result.manifest["coordinateSystem"] == "threejs-right-handed-y-up"


def test_loader_error_is_sanitized(tmp_path: Path) -> None:
    source = tmp_path / "bad.pt"
    output = tmp_path / "bad.omx4d"
    source.write_bytes(b"not a torch archive and /private/attacker/path")
    with pytest.raises(ConversionError) as caught:
        convert_pt_file(source, output)
    assert caught.value.code == "invalid_pt_archive"
    assert "/private/attacker/path" not in caught.value.message


def test_api_streams_binary_and_removes_temporary_files(
    tmp_path: Path, valid_predictions: dict[str, torch.Tensor]
) -> None:
    source = tmp_path / "fixture.pt"
    torch.save(valid_predictions, source)
    upload = source.read_bytes()
    temp_directory = tmp_path / "service-temp"
    temp_directory.mkdir()
    settings = ServerSettings(
        limits=IngestionLimits(max_upload_bytes=10 * 1024 * 1024),
        conversion_timeout_seconds=30,
        temp_directory=str(temp_directory),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/convert",
            files={"file": ("my predictions.pt", upload, "application/octet-stream")},
            data={"point_budget": "7", "fps": "12"},
        )
    assert response.status_code == 200
    assert response.content.startswith(MAGIC)
    assert response.headers["x-omnix-point-count"] == "7"
    assert not list(temp_directory.iterdir())


def test_api_returns_safe_schema_error(
    tmp_path: Path, valid_predictions: dict[str, torch.Tensor]
) -> None:
    valid_predictions["extra"] = torch.tensor(1.0)
    source = tmp_path / "wrong.pt"
    torch.save(valid_predictions, source)
    settings = ServerSettings(temp_directory=str(tmp_path))
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/convert",
            files={"file": ("wrong.pt", source.read_bytes())},
            data={"point_budget": "7", "fps": "15"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_schema_keys"
    assert "traceback" not in response.text.lower()
