from __future__ import annotations

import torch
import pytest


@pytest.fixture
def valid_predictions() -> dict[str, torch.Tensor]:
    source_views, frames, height, width = 2, 3, 3, 4
    trajectory = torch.empty(
        (source_views, frames, height, width, 3), dtype=torch.float32
    )
    for source in range(source_views):
        for frame in range(frames):
            for y in range(height):
                for x in range(width):
                    trajectory[source, frame, y, x] = torch.tensor(
                        [x + frame * 0.25, y + source * 5.0, 2.0 + source],
                        dtype=torch.float32,
                    )

    camera_pose = torch.zeros((source_views, 3, 4), dtype=torch.float32)
    camera_pose[:, :3, :3] = torch.eye(3, dtype=torch.float32)
    camera_pose[0, :, 3] = torch.tensor([1.0, 2.0, 3.0])
    camera_pose[1, :, 3] = torch.tensor([-1.0, 4.0, 2.0])
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).repeat(source_views, 1, 1)
    dynamic_score = torch.linspace(
        0.0, 1.0, source_views * height * width, dtype=torch.float32
    ).reshape(source_views, height, width)
    return {
        "trajectory": trajectory,
        "camera_pose": camera_pose,
        "intrinsics": intrinsics,
        "pts3d_dynamic_score": dynamic_score,
    }
