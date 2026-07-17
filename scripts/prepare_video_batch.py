#!/usr/bin/env python3
"""Prepare fixed-rate, non-overlapping image chunks for OmniX batch inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must stay inside the repository: {value}") from error
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def docker_media_command(
    image: str, entrypoint: str, arguments: list[str]
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{REPO_ROOT}:/workspace/OmniX",
        "--workdir",
        "/workspace/OmniX",
        "--entrypoint",
        entrypoint,
        image,
        *arguments,
    ]


def probe_video(image: str, source: str) -> dict[str, Any]:
    command = docker_media_command(
        image,
        "ffprobe",
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration",
            "-of",
            "json",
            source,
        ],
    )
    result = subprocess.run(
        command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)["streams"][0]


def extract_frames(
    image: str,
    source: str,
    destination: Path,
    fps: float,
    crop_filter: str | None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for old_frame in destination.glob("frame_*.jpg"):
        old_frame.unlink()

    filters = [crop_filter] if crop_filter else []
    filters.append(f"fps={fps:g}")
    output_pattern = f"{relative(destination)}/frame_%06d.jpg"
    command = docker_media_command(
        image,
        "ffmpeg",
        [
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            source,
            "-vf",
            ",".join(filters),
            "-start_number",
            "0",
            "-q:v",
            "2",
            output_pattern,
        ],
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_link(source: Path, destination: Path) -> None:
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_video(
    video: dict[str, Any],
    output_root: Path,
    image: str,
    fps: float,
    chunk_frames: int,
    force_extract: bool,
) -> dict[str, Any]:
    video_id = video["id"]
    source = repo_path(video["source"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing source video: {source}")

    source_digest = sha256(source)
    sampled_dir = output_root / "input" / "sampled" / video_id
    sampled_marker = sampled_dir / "sampled_manifest.json"
    crop_filter = video.get("crop_filter")
    expected_marker = {
        "source": relative(source),
        "source_sha256": source_digest,
        "sampling_fps": fps,
        "crop_filter": crop_filter,
    }

    sampled_frames = sorted(sampled_dir.glob("frame_*.jpg"))
    marker_matches = False
    if sampled_marker.is_file() and sampled_frames and not force_extract:
        marker = json.loads(sampled_marker.read_text())
        marker_matches = all(
            marker.get(key) == value for key, value in expected_marker.items()
        )
        marker_matches = marker_matches and marker.get("sampled_frame_count") == len(
            sampled_frames
        )

    if not marker_matches:
        print(f"Extracting {video_id} at {fps:g} fps")
        extract_frames(image, relative(source), sampled_dir, fps, crop_filter)
        sampled_frames = sorted(sampled_dir.glob("frame_*.jpg"))
        if not sampled_frames:
            raise RuntimeError(f"FFmpeg produced no frames for {video_id}")
        atomic_json(
            sampled_marker,
            {
                **expected_marker,
                "sampled_frame_count": len(sampled_frames),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    else:
        print(f"Reusing {len(sampled_frames)} sampled frames for {video_id}")

    probe = probe_video(image, relative(source))
    chunk_count = math.ceil(len(sampled_frames) / chunk_frames)
    chunks: list[dict[str, Any]] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_frames
        end = min(start + chunk_frames, len(sampled_frames))
        valid_frames = end - start
        pad_frames = chunk_frames - valid_frames
        chunk_id = f"chunk_{chunk_index:04d}"
        chunk_root = output_root / "input" / "chunks" / video_id / chunk_id
        frame_dir = chunk_root / "video_00"
        frame_dir.mkdir(parents=True, exist_ok=True)

        source_indices = list(range(start, end))
        source_indices.extend([end - 1] * pad_frames)
        expected_names = {f"frame_{index:04d}.jpg" for index in range(chunk_frames)}
        for stale_frame in frame_dir.glob("frame_*.jpg"):
            if stale_frame.name not in expected_names:
                stale_frame.unlink()
        for local_index, source_index in enumerate(source_indices):
            ensure_link(
                sampled_frames[source_index], frame_dir / f"frame_{local_index:04d}.jpg"
            )

        start_ms = round(start * 1000 / fps)
        end_ms = round(end * 1000 / fps)
        output_dir = output_root / "output" / video_id / chunk_id
        pt_name = (
            f"{video_id}__fps{fps:g}__chunk-{chunk_index:04d}__"
            f"t-{start_ms:09d}-{end_ms:09d}ms__valid{valid_frames:02d}-pad{pad_frames:02d}.pt"
        )
        chunks.append(
            {
                "id": chunk_id,
                "index": chunk_index,
                "input_dir": relative(chunk_root),
                "output_dir": relative(output_dir),
                "prediction_file": relative(output_dir / pt_name),
                "sample_start_index": start,
                "sample_end_index_exclusive": end,
                "source_sample_indices": source_indices,
                "sample_times_s": [round(index / fps, 6) for index in source_indices],
                "start_time_s": round(start / fps, 6),
                "end_time_s": round(end / fps, 6),
                "valid_frames": valid_frames,
                "pad_frames": pad_frames,
            }
        )

    chunks_root = output_root / "input" / "chunks" / video_id
    expected_chunk_dirs = {chunk["id"] for chunk in chunks}
    if chunks_root.is_dir():
        for stale_chunk in chunks_root.iterdir():
            if stale_chunk.is_dir() and stale_chunk.name.startswith("chunk_"):
                if stale_chunk.name not in expected_chunk_dirs:
                    shutil.rmtree(stale_chunk)

    return {
        "id": video_id,
        "source": relative(source),
        "source_sha256": source_digest,
        "source_probe": probe,
        "crop_filter": crop_filter,
        "sampled_frames_dir": relative(sampled_dir),
        "sampled_frame_count": len(sampled_frames),
        "chunk_count": chunk_count,
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", required=True, help="JSON batch specification inside the repo"
    )
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    spec_path = repo_path(args.spec)
    spec = json.loads(spec_path.read_text())
    if spec.get("schema") != "omnix.video-batch-spec.v1":
        raise ValueError(f"Unsupported batch spec schema: {spec.get('schema')!r}")
    fps = float(spec["sampling_fps"])
    chunk_frames = int(spec["frames_per_chunk"])
    if fps <= 0 or chunk_frames <= 0:
        raise ValueError("sampling_fps and frames_per_chunk must be positive")

    output_root = repo_path(spec["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    image = spec.get("docker_image", "omnix-dgx-spark:latest")
    video_ids = [video["id"] for video in spec["videos"]]
    invalid_ids = [
        video_id
        for video_id in video_ids
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", video_id) is None
    ]
    if invalid_ids:
        raise ValueError(f"Video IDs must be lowercase slugs: {invalid_ids}")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("Video IDs must be unique")
    videos = [
        prepare_video(video, output_root, image, fps, chunk_frames, args.force_extract)
        for video in spec["videos"]
    ]

    checkpoint = repo_path(
        spec.get("checkpoint", "pretrained_weight/eccv_release.ckpt")
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format={{.Id}}", image],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest = {
        "schema": "omnix.batch-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": relative(spec_path),
        "output_root": relative(output_root),
        "sampling": {
            "fps": fps,
            "frames_per_chunk": chunk_frames,
            "non_overlapping": True,
            "tail_policy": "repeat-last-frame-padding",
            "model_width": 504,
            "model_height": 280,
        },
        "provenance": {
            "git_commit": git_commit,
            "docker_image": image,
            "docker_image_id": image_id,
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "videos": videos,
        "total_chunks": sum(video["chunk_count"] for video in videos),
    }
    manifest_path = output_root / "batch_plan.json"
    atomic_json(manifest_path, manifest)
    print(
        f"Prepared {manifest['total_chunks']} chunks; wrote {relative(manifest_path)}"
    )


if __name__ == "__main__":
    main()
