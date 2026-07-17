"""FastAPI entry point for bounded local `.pt` conversion."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ServerSettings
from .converter import (
    ConversionControl,
    ConversionOptions,
    convert_pt_file,
    sanitize_dataset_name,
)
from .errors import ConversionError, ResourceLimitError


def _unlink(path: str | os.PathLike[str] | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


async def _stream_upload(
    upload: UploadFile, destination: Path, settings: ServerSettings
) -> int:
    total = 0
    try:
        with destination.open("wb") as file:
            while True:
                chunk = await upload.read(settings.upload_chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.limits.max_upload_bytes:
                    raise ResourceLimitError(
                        "upload_too_large",
                        "The uploaded file exceeds the configured size limit.",
                        details={
                            "receivedAtLeast": total,
                            "limit": settings.limits.max_upload_bytes,
                        },
                    )
                file.write(chunk)
    except OSError as exc:
        raise ConversionError(
            "upload_storage_error",
            "The service could not store the upload for conversion.",
            status_code=507,
        ) from exc
    finally:
        await upload.close()
    if total == 0:
        raise ConversionError("empty_upload", "The uploaded file is empty.", status_code=400)
    return total


def _temporary_path(settings: ServerSettings, suffix: str) -> Path:
    descriptor, value = tempfile.mkstemp(
        prefix="omnix-", suffix=suffix, dir=settings.temp_directory
    )
    os.close(descriptor)
    return Path(value)


def _stream_and_delete(path: Path) -> Iterator[bytes]:
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                yield chunk
    finally:
        _unlink(path)


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    app = FastAPI(
        title="OmniX visualizer ingestion",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(
        _request: Request, exc: ConversionError
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "schemaVersion": 1,
            "maxUploadBytes": settings.limits.max_upload_bytes,
            "maxPointBudget": settings.limits.max_point_budget,
        }

    @app.post("/api/convert")
    async def convert(
        file: UploadFile = File(...),
        point_budget: int = Form(100_000),
        fps: float = Form(15.0),
    ) -> StreamingResponse:
        source_path = _temporary_path(settings, ".pt")
        output_path = _temporary_path(settings, ".omx4d")
        deferred_cleanup = False
        response_owns_output = False
        worker: asyncio.Task[object] | None = None
        cancel_event = threading.Event()
        try:
            await _stream_upload(file, source_path, settings)
            options = ConversionOptions(
                point_budget=point_budget,
                fps=fps,
                name=sanitize_dataset_name(file.filename),
            )
            deadline = time.monotonic() + settings.conversion_timeout_seconds
            worker = asyncio.create_task(
                asyncio.to_thread(
                    convert_pt_file,
                    source_path,
                    output_path,
                    options=options,
                    limits=settings.limits,
                    control=ConversionControl(
                        deadline=deadline, cancelled=cancel_event
                    ),
                )
            )
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=settings.conversion_timeout_seconds,
                )
            except TimeoutError as exc:
                cancel_event.set()
                deferred_cleanup = True
                worker.add_done_callback(
                    lambda _task: (_unlink(source_path), _unlink(output_path))
                )
                raise ConversionError(
                    "conversion_timeout",
                    "Conversion exceeded the configured wall-clock limit.",
                    status_code=408,
                ) from exc

            output_size = output_path.stat().st_size
            manifest = result.manifest  # type: ignore[union-attr]
            response_owns_output = True
            return StreamingResponse(
                _stream_and_delete(output_path),
                media_type="application/vnd.omnix.omx4d",
                headers={
                    "Content-Length": str(output_size),
                    "Content-Disposition": 'attachment; filename="dataset.omx4d"',
                    "X-OmniX-Schema-Version": str(manifest["schemaVersion"]),
                    "X-OmniX-Point-Count": str(manifest["pointCount"]),
                    "X-OmniX-Frame-Count": str(manifest["frameCount"]),
                },
            )
        except asyncio.CancelledError:
            cancel_event.set()
            if worker is not None and not worker.done():
                deferred_cleanup = True
                worker.add_done_callback(
                    lambda _task: (_unlink(source_path), _unlink(output_path))
                )
            raise
        finally:
            if not deferred_cleanup:
                _unlink(source_path)
                if not response_owns_output:
                    _unlink(output_path)

    return app


app = create_app()
