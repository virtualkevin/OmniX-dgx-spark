"""Load optional companion RGB frames for fixture generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .errors import ConversionError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_source_rgb(image_directory: str | Path) -> torch.Tensor:
    """Return sorted RGB images as contiguous uint8 ``[N,H,W,3]`` data."""

    directory = Path(image_directory)
    if not directory.is_dir():
        raise ConversionError(
            "invalid_image_directory",
            "The companion image directory does not exist.",
            status_code=400,
        )
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not paths:
        raise ConversionError(
            "missing_source_images",
            "The companion image directory has no supported RGB images.",
            status_code=400,
        )

    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int, int] | None = None
    for path in paths:
        try:
            with Image.open(path) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise ConversionError(
                "invalid_source_image",
                f"Companion image '{path.name}' could not be decoded.",
                status_code=400,
            ) from exc
        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ConversionError(
                "inconsistent_source_images",
                "All companion images must have the same dimensions.",
                status_code=400,
            )
        frames.append(frame)

    return torch.from_numpy(np.stack(frames)).contiguous()
