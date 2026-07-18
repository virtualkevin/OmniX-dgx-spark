#!/usr/bin/env python3
"""Render one source view from an existing OmniX prediction PT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from visualize_simple import (
    IMG_EXTS,
    crop_to_resolution,
    ensure_4x4,
    move_cameras_backwards,
    visualize_3d_trajectories,
)


def image_paths(image_root: Path) -> list[Path]:
    candidates = [image_root / "video_00", image_root]
    for candidate in candidates:
        if candidate.is_dir():
            paths = sorted(
                path
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMG_EXTS
            )
            if paths:
                return paths
    raise FileNotFoundError(f"No input frames found below {image_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-view", type=int, default=16)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--orbit-frames", type=int, default=12)
    parser.add_argument("--render-scale", type=int, default=1)
    args = parser.parse_args()

    raw = torch.load(
        args.prediction, map_location="cpu", mmap=True, weights_only=True
    )
    trajectory = raw["trajectory"]
    source_views, frames, height, width, _ = trajectory.shape
    if not 0 <= args.source_view < source_views:
        raise ValueError(
            f"--source-view must be in [0, {source_views}), got {args.source_view}"
        )
    paths = image_paths(args.image_root)
    if len(paths) != source_views:
        raise RuntimeError(f"Found {len(paths)} images; expected {source_views}")
    with PIL.Image.open(paths[args.source_view]) as image:
        image = crop_to_resolution(image.convert("RGB"), (width, height))
        source_rgb = np.asarray(image, dtype=np.float32)[None, ...] / 255.0

    source_slice = slice(args.source_view, args.source_view + 1)
    trajectories = trajectory[source_slice].numpy()
    camera_pose = ensure_4x4(raw["camera_pose"][source_slice].numpy())
    camera_pose = move_cameras_backwards(
        camera_pose, distance=0.16, lift_height=0.1, tilt_down_deg=-6
    )
    intrinsics = raw["intrinsics"][source_slice].numpy()
    dynamic_score = raw["pts3d_dynamic_score"][source_slice].numpy()
    dynamic_points = int((dynamic_score > 0.5).sum())
    track_count = max(64, min(2_000, dynamic_points // 100))

    np.random.seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_3d_trajectories(
        trajectories=trajectories,
        intrinsics=intrinsics,
        c2ws=camera_pose,
        foreground_masks=dynamic_score,
        image_array=source_rgb,
        save_dir=args.output_dir,
        num_points_to_show=track_count,
        traj_length=frames,
        traj_alpha=1.0,
        traj_width=2,
        img_h=height * args.render_scale,
        img_w=width * args.render_scale,
        fps=args.fps,
        show_dense_point_cloud=True,
        project_all_views=False,
        point_size=max(1, args.render_scale),
        orbit_frames=args.orbit_frames,
        orbit_radius=0.1,
    )
    generated = args.output_dir / "video_cam_00.mp4"
    final_path = args.output_dir / (
        f"{args.prediction.stem}__view-{args.source_view:02d}"
        f"__{args.fps:g}fps.mp4"
    )
    os.replace(generated, final_path)
    print(final_path)


if __name__ == "__main__":
    main()
