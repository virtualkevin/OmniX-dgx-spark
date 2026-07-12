import json
import shutil
import os
import argparse
import traceback
from typing import Dict
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# -------------------------- 配置路径 --------------------------
SCENE_ROOT =  "/cos_nj1/share_303535725/hunyuan_world/4d_data/omniworld_unzip/omnigame/"
TARGET_ROOT = "/cos_nj1/share_303535725/yanqinjiang/processed_data/omnigame"
NUM_PROCESSES = int(cpu_count() * 0.8)  # 进程数量，根据机器核心数调整，默认为16
# -------------------------------------------------------------

def load_split_info(scene_dir: Path):
    """Return the split json dict."""
    split_path = scene_dir / "split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split info not found: {split_path}")
        
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_poses(scene_dir: Path, split_info: Dict, split_idx: int):
    """
    加载相机参数并转换为 c2w
    """
    # Key in JSON usually is string, e.g. "0", "1"
    try:
        frame_idxs = split_info["split"][split_idx]
    except KeyError:
        frame_idxs = split_info["split"][str(split_idx)]
        
    frame_count = len(frame_idxs)

    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    # ----- intrinsics --------------------------------------------------------
    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0)
    intrinsics[:, 0, 0] = cam["focals"]          # fx
    intrinsics[:, 1, 1] = cam["focals"]          # fy
    intrinsics[:, 0, 2] = cam["cx"]              # cx
    intrinsics[:, 1, 2] = cam["cy"]              # cy

    # ----- extrinsics --------------------------------------------------------
    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)

    # SciPy expects quaternions as (x, y, z, w)
    quat_wxyz = np.array(cam["quats"])           # (S, 4)  (w,x,y,z)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)

    # 1. 获取原始的 W2C 旋转和平移
    rotations_w2c = R.from_quat(quat_xyzw).as_matrix()  # (S, 3, 3)
    translations_w2c = np.array(cam["trans"])           # (S, 3)

    # 2. 计算 C2W (求逆)
    rotations_c2w = rotations_w2c.transpose(0, 2, 1)
    
    # 平移向量求逆公式：t_c2w = - (R_c2w @ t_w2c)
    t_vec = translations_w2c[:, :, None] 
    translations_c2w = -np.matmul(rotations_c2w, t_vec)
    translations_c2w = translations_c2w[:, :, 0]

    # 3. 赋值
    extrinsics[:, :3, :3] = rotations_c2w
    extrinsics[:, :3, 3] = translations_c2w
    
    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)


def process_single_scene(args):
    """
    单个场景的处理函数，用于多进程调用
    """
    scene_name, scene_root_str, target_root_str = args
    
    scene_path = Path(scene_root_str) / scene_name
    target_root_path = Path(target_root_str)

    try:
        # 1. Load Split Info
        split_info = load_split_info(scene_path)
        split_num = split_info["split_num"]
        
        # 2. Check source folders (Optional)
        # 简单的存在性检查，不在这里做严格assert以免中断其他逻辑
        rgb_folder = scene_path / "color"
        if not rgb_folder.exists():
            return f"Skipped {scene_name}: Color folder missing"

        # 3. Process Splits
        for split_idx in range(split_num):
            output_dir = target_root_path / f"{scene_name}_{split_idx:03d}"
            
            # Create subdirectories
            (output_dir / "image" / "cam_0").mkdir(parents=True, exist_ok=True)
            (output_dir / "depth" / "cam_0").mkdir(parents=True, exist_ok=True)
            (output_dir / "foreground_mask" / "cam_0").mkdir(parents=True, exist_ok=True)
            (output_dir / "camera" / "cam_0").mkdir(parents=True, exist_ok=True)

            # Get frame indices
            try:
                frame_idxs = split_info["split"][split_idx]
            except KeyError:
                frame_idxs = split_info["split"][str(split_idx)]
        
            # Load camera poses
            K, c2w = load_camera_poses(scene_path, split_info=split_info, split_idx=split_idx)
            
            if K.shape[0] != len(frame_idxs):
                print(f"[{scene_name}] Error: Split {split_idx} pose count mismatch.")
                continue

            # Process frames
            # 注意：在多进程中不要使用 tqdm(frame_idxs)，否则控制台会非常混乱
            for new_idx, frame_idx in enumerate(frame_idxs):
                frame_name = f"{frame_idx:06d}.png" 

                # Source Paths
                rgb_src = scene_path / "color" / frame_name
                depth_src = scene_path / "depth" / frame_name
                mask_src = scene_path / "gdino_mask" / frame_name
                
                # Target Paths
                rgb_dst = output_dir / "image" / "cam_0" / f"frame_{new_idx:04d}.png"
                depth_dst = output_dir / "depth" / "cam_0" / f"frame_{new_idx:04d}.png"
                mask_dst = output_dir / "foreground_mask" / "cam_0" / f"frame_{new_idx:04d}.png"
                cam_dst = output_dir / "camera" / "cam_0" / f"frame_{new_idx:04d}.npz"

                # Copy Operations
                if rgb_src.exists(): shutil.copy(rgb_src, rgb_dst)
                if depth_src.exists(): shutil.copy(depth_src, depth_dst)
                if mask_src.exists(): shutil.copy(mask_src, mask_dst)

                # Save Camera
                intrinsic = K[new_idx]
                camera_pose = c2w[new_idx]
                np.savez(cam_dst, intrinsic=intrinsic, camera_pose=camera_pose)
        
        return None # Success

    except Exception as e:
        # 捕获所有异常，防止单个场景导致程序崩溃，并返回错误信息
        error_msg = f"Error processing {scene_name}: {str(e)}{traceback.format_exc()}"
        return error_msg


def main():
    scene_root = Path(SCENE_ROOT)
    target_root = Path(TARGET_ROOT)
    
    if not scene_root.exists():
        print(f"Error: Scene root does not exist: {scene_root}")
        return

    # 1. 获取所有场景名称 (只处理文件夹)
    print(f"Scanning scenes in {scene_root}...")
    scene_names = [p.name for p in scene_root.iterdir() if p.is_dir()]
    scene_names.sort()
    # tmp_debug
    with open("/apdcephfs/private_yanqinjiang/project/dream4d/preprocess/omnigame/missing_paths.txt", "r") as file_to_read:
        lines = file_to_read.readlines()
    scene_names = [line.strip().split("	")[0] for line in lines[1:]]
    scene_names = list(set(scene_names))
    
    total_scenes = len(scene_names)
    print(f"Found {total_scenes} scenes.")

    # 2. 准备多进程参数
    # 每个任务的参数是一个tuple: (scene_name, str(scene_root), str(target_root))
    # 传递字符串路径给子进程更安全（Pickle兼容性更好）
    process_args = [(name, str(scene_root), str(target_root)) for name in scene_names]

    # 3. 启动多进程池
    print(f"Starting processing with {NUM_PROCESSES} processes...")
    
    error_logs = []
    
    # 使用 imap_unordered 可以让进度条更流畅，配合 tqdm
    with Pool(processes=NUM_PROCESSES) as pool:
        # tqdm 负责显示总的场景进度
        results = list(tqdm(pool.imap_unordered(process_single_scene, process_args), total=total_scenes, desc="Processing Scenes"))
        
        # 收集错误日志
        for res in results:
            if res is not None:
                error_logs.append(res)

    # 4. 打印总结
    print("" + "="*50)
    print("Processing Complete.")
    if error_logs:
        print(f"Identified {len(error_logs)} errors:")
        for log in error_logs:
            print("-" * 20)
            print(log)
    else:
        print("All scenes processed successfully without exceptions.")
    print("="*50)


if __name__ == "__main__":
    main()