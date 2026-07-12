#!/usr/bin/env python3
"""
Preprocess the Dynamic Replica dataset.
Refactored for sequence-level parallelism.
"""

import argparse
import gzip
import json
import os
import os.path as osp
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)
from tqdm import tqdm

# Enable OpenEXR support in OpenCV.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# --- Helper Functions (No changes here) ---

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    camera_name: Optional[str] = None
    instance_id_map_path: Optional[str] = None
    flow_forward: Optional[str] = None
    flow_forward_mask: Optional[str] = None
    flow_backward: Optional[str] = None
    flow_backward_mask: Optional[str] = None
    trajectories: Optional[str] = None

def _get_pytorch3d_camera(entry_viewpoint, image_size, scale: float):
    assert entry_viewpoint is not None
    principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    fmt = entry_viewpoint.intrinsics_format
    if fmt.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig
    elif fmt.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {fmt}")

    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
    T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    T_pytorch3d[..., :2] *= -1
    R_pytorch3d[..., :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d
    R = R.transpose(-2, -1) 

    return R, tvec, focal_length_px, principal_point_px

# --- Core Logic Refactored ---

def gather_tasks_for_split(split, root_dir, out_dir):
    """
    Reads the JSON for a split and returns a list of tasks.
    Each task contains all frames for a specific (sequence, camera).
    """
    split_dir = osp.join(root_dir, split)
    frame_annotations_file = osp.join(split_dir, f"frame_annotations_{split}.jgz")
    
    print(f"Loading annotations for split '{split}' from {frame_annotations_file}...")
    with gzip.open(frame_annotations_file, "rt", encoding="utf8") as zipfile:
        frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])

    # Group frames by sequence and camera.
    # Structure: seq_annot[seq_name][cam_name] = [List of Frames]
    seq_annot = defaultdict(lambda: defaultdict(list))
    for frame_annot in frame_annots_list:
        # # tmp_debug
        # if frame_annot.sequence_name not in ["273e91-4_obj"]:
        #     continue
        seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(frame_annot)

    tasks = []
    # Flatten the dictionary into a list of tasks
    for seq_name, cam_dict in seq_annot.items():
        for cam_name, frames in cam_dict.items():
            # Create a task object/tuple
            task = {
                "split": split,
                "root_dir": root_dir,
                "out_dir": out_dir,
                "seq_name": seq_name,
                "cam_name": cam_name,
                "frames": frames
            }
            tasks.append(task)
    
    return tasks

def process_single_sequence_task(task):
    """
    Worker function: Processes all frames for ONE sequence and ONE camera view.
    """
    split = task["split"]
    root_dir = task["root_dir"]
    out_dir = task["out_dir"]
    seq_name = task["seq_name"]
    cam_name = task["cam_name"]
    frames = task["frames"]
    
    split_dir = osp.join(root_dir, split)

    # Output directories setup
    out_img_dir = osp.join(out_dir, split, seq_name, cam_name, "rgb")
    out_depth_dir = osp.join(out_dir, split, seq_name, cam_name, "depth")
    out_mask_dir = osp.join(out_dir, split, seq_name, cam_name, "foreground_mask")
    out_traj_dir = osp.join(out_dir, split, seq_name, cam_name, "traj")
    out_cam_dir = osp.join(out_dir, split, seq_name, cam_name, "cam")
    
    # Create dirs (exist_ok=True handles race conditions if multiple cams processed same seq dir structure, 
    # though here we separate by cam folder anyway)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_traj_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    # Process each frame in this sequence/camera
    for framedata in frames:
        frame_number = framedata.frame_number
        
        # Paths construction
        im_path = osp.join(split_dir, framedata.image.path)
        depth_path = osp.join(split_dir, framedata.depth.path)
        mask_path = osp.join(split_dir, framedata.mask.path)
        
        trajectory_path = None
        if framedata.trajectories and framedata.trajectories["path"]:
            trajectory_path = osp.join(split_dir, framedata.trajectories["path"])

        # Basic validations
        if not os.path.isfile(im_path): continue # Or raise error
        if not os.path.isfile(depth_path): continue
        if not os.path.isfile(mask_path): continue
            
        # 1. Load Depth
        depth = _load_16big_png_depth(depth_path)

        # 2. Process Trajectory (if exists)
        if trajectory_path and os.path.isfile(trajectory_path):
            try:
                trajectory = torch.load(trajectory_path, weights_only=True)
                save_path = osp.join(out_traj_dir, f"{frame_number:04d}.npz")
                np.savez(
                    save_path, 
                    traj_3d_world=trajectory["traj_3d_world"].numpy(),
                )
            except Exception as e:
                print(f"Error loading trajectory for {seq_name} {frame_number}: {e}")

        # 3. Process Camera
        viewpoint = framedata.viewpoint
        R, t, focal, pp = _get_pytorch3d_camera(
            viewpoint, framedata.image.size, scale=1.0
        )
        intrinsics = np.eye(3)
        intrinsics[0, 0] = focal[0].item()
        intrinsics[1, 1] = focal[1].item()
        intrinsics[0, 2] = pp[0].item()
        intrinsics[1, 2] = pp[1].item()
        
        pose = np.eye(4)
        pose[:3, :3] = R.numpy().T
        pose[:3, 3] = -R.numpy().T @ t.numpy()

        # 4. Save Outputs
        out_img_path = osp.join(out_img_dir, f"{frame_number:04d}.png")
        out_depth_path = osp.join(out_depth_dir, f"{frame_number:04d}.npy")
        out_mask_path = osp.join(out_mask_dir, f"{frame_number:04d}.png")
        out_cam_path = osp.join(out_cam_dir, f"{frame_number:04d}.npz")
        
        shutil.copy(im_path, out_img_path)
        np.save(out_depth_path, depth)
        shutil.copy(mask_path, out_mask_path)
        np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)

    return f"{split}/{seq_name}/{cam_name} Done"

# Global config
SPLITS = ["train", "valid", "test"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=SPLITS)
    parser.add_argument("--num_processes", type=int, default=int(cpu_count() * 0.8))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Gather all tasks first
    # We do this sequentially because loading the big JSON is memory intensive 
    # and reading from same zip in parallel processes is tricky/inefficient.
    all_tasks = []
    for split in args.splits:
        tasks = gather_tasks_for_split(split, args.root_dir, args.out_dir)
        print(f"Split '{split}': Found {len(tasks)} (Sequence+Camera) tasks.")
        all_tasks.extend(tasks)

    print(f"Total tasks to process: {len(all_tasks)}")
    print(f"Starting multiprocessing with {args.num_processes} processes...")

    # 2. Process tasks in parallel
    # Chunksize determines how many sequences a worker grabs at once.
    # 1 is fine for load balancing, slightly higher (e.g., 5) reduces IPC overhead.
    with Pool(processes=args.num_processes) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_single_sequence_task, all_tasks, chunksize=1),
                total=len(all_tasks),
                desc="Processing Sequences"
            )
        )

if __name__ == "__main__":
    main()