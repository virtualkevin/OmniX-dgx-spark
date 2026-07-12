import os
import argparse
import numpy as np
import json
import cv2
import h5py
import shutil
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ================= 配置常量 =================
# 注意：读取时请记得除以 65535.0 并乘以 10.0 (如果你之后要转 metric)
# 这里仅作搬运，不涉及数值变换
# 使用
# python preprocess_droid.py --raw_root "/apdcephfs_jn3/share_303535725/yanqinjiang/others/omniworld_droid_raw/annotations/OmniWorld-DROID/droid_processed/1.0.1" --anno_root "/apdcephfs_jn3/share_303535725/yanqinjiang/others/omniworld_droid_raw/annotations/OmniWorld-DROID/droid_processed/1.0.1" --output_root "/apdcephfs_jn3/share_303535725/yanqinjiang/others/omniworld_droid_debug/" --sequence_path "TRI/success/2023-10-17/Tue_Oct_17_17:20:55_2023"
def get_c2w_matrix(vec6):
    """
    将 6D 向量 [tx, ty, tz, rx, ry, rz] 转换为 4x4 C2W 矩阵
    假设旋转顺序为 Euler XYZ
    """
    vec6 = np.array(vec6)
    mat = np.eye(4)
    mat[:3, 3] = vec6[:3] # Translation
    
    # Rotation (Euler XYZ)
    rot_euler = vec6[3:]
    rot_matrix = Rotation.from_euler("xyz", rot_euler).as_matrix()
    mat[:3, :3] = rot_matrix
    
    return mat

def process_camera_view(cam_name, serial, paths, data_cache, args):
    """
    处理单个相机视角的逻辑
    """
    # 1. 准备输出路径
    # 结构: output_root/sequence/camera_name/{image, depth, camera}
    cam_output_root = os.path.join(args.output_root, args.sequence_path, cam_name)
    
    out_img_dir = os.path.join(cam_output_root, "image")
    out_depth_dir = os.path.join(cam_output_root, "depth")
    out_cam_dir = os.path.join(cam_output_root, "camera")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    # 2. 准备输入数据源
    video_path = os.path.join(paths['video_root'], f"{serial}.mp4")
    depth_folder = os.path.join(paths['depth_root'], serial)
    
    # 检查视频是否存在
    if not os.path.exists(video_path):
        print(f"[Warning] Video not found for {cam_name}: {video_path}")
        return

    # 3. 打开视频读取 RGB
    cap = cv2.VideoCapture(video_path)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 4. 获取 Trajectory 数据
    traj_data = data_cache['trajectories'].get(cam_name)
    if traj_data is None:
        print(f"[Warning] No trajectory data for {cam_name}")
        cap.release()
        return
    
    # 5. 获取 Intrinsics
    K = data_cache['intrinsics'].get(cam_name)

    # 6. 计算有效帧数 (取最小值，防止索引越界)
    # 检查深度图文件夹里的文件数
    if os.path.exists(depth_folder):
        depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
        depth_count = len(depth_files)
    else:
        depth_count = 0
        print(f"[Warning] Depth folder not found: {depth_folder}")

    # 对齐帧数：取 视频帧数、轨迹长度、深度图数量 的最小值
    valid_frames = min(video_frame_count, len(traj_data), depth_count)
    
    print(f"Processing {cam_name} (Serial: {serial}) | Aligned Frames: {valid_frames}")

    for i in tqdm(range(valid_frames), desc=f"  > {cam_name}"):
        # --- A. 读取/保存 RGB ---
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存 JPG
        img_save_path = os.path.join(out_img_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(img_save_path, frame)
        
        # --- B. 处理/保存 Depth (PNG Direct Copy) ---
        # 原始文件名: 0.png, 1.png (无前导零)
        depth_src_path = os.path.join(depth_folder, f"{i}.png")
        depth_dst_path = os.path.join(out_depth_dir, f"frame_{i:04d}.png")
        
        if os.path.exists(depth_src_path):
            # 直接复制文件 (IO 效率最高，保留原始 16bit)
            shutil.copy(depth_src_path, depth_dst_path)
        else:
            # 如果中间缺帧，生成全黑的 16bit PNG 占位
            H, W = frame.shape[:2]
            dummy_depth = np.zeros((H, W), dtype=np.uint16)
            cv2.imwrite(depth_dst_path, dummy_depth)
        
        # --- C. 处理/保存 Camera Pose ---
        # 获取当前帧的外参向量
        vec6 = traj_data[i]
        c2w = get_c2w_matrix(vec6)
        
        # 保存 NPZ (keys: intrinsics, camera_pose)
        cam_save_path = os.path.join(out_cam_dir, f"frame_{i:04d}.npz")
        np.savez(cam_save_path, intrinsics=K, camera_pose=c2w)

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Preprocess DROID dataset")
    
    # 路径参数
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Root directory of the RAW data (containing recordings/MP4)")
    
    parser.add_argument("--anno_root", type=str, required=True,
                        help="Root directory of the ANNOTATIONS (containing recordings/trajectory.h5, foundation_stereo)")
    
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for output")
    
    parser.add_argument("--sequence_path", type=str, required=True,
                        help="Relative path to the specific sequence (e.g. TRI/success/...)")
    
    args = parser.parse_args()

    # 构建完整路径
    # Raw Data Path (Video)
    raw_seq_path = os.path.join(args.raw_root, args.sequence_path)
    # Annotation Data Path (Meta, Traj, Depth)
    anno_seq_path = os.path.join(args.anno_root, args.sequence_path)
    
    paths = {
        'meta': os.path.join(anno_seq_path, "meta_info.json"),
        'intrinsics': os.path.join(anno_seq_path, "recordings", "camera_info_dict.npy"),
        'trajectory': os.path.join(anno_seq_path, "recordings", "trajectory.h5"),
        'depth_root': os.path.join(anno_seq_path, "foundation_stereo"),
        
        # Video 在 Raw Root 下
        'video_root': os.path.join(raw_seq_path, "recordings", "MP4") 
    }

    # 1. 加载元数据 (Meta Info)
    if not os.path.exists(paths['meta']):
        print(f"Meta file not found: {paths['meta']}")
        return

    with open(paths['meta'], "r") as f:
        meta_info = json.load(f)

    serials = {
        "wrist": meta_info["wrist_cam_serial"],
        "ext1": meta_info["ext1_cam_serial"],
        "ext2": meta_info["ext2_cam_serial"]
    }

    # 2. 加载内参 (Intrinsics)
    if not os.path.exists(paths['intrinsics']):
        print(f"Intrinsics file not found: {paths['intrinsics']}")
        return
        
    camera_intrinsic_info = np.load(paths['intrinsics'], allow_pickle=True).item()
    intrinsics_cache = {
        "wrist": camera_intrinsic_info[serials["wrist"]]["cam_matrix"],
        "ext1": camera_intrinsic_info[serials["ext1"]]["cam_matrix"],
        "ext2": camera_intrinsic_info[serials["ext2"]]["cam_matrix"]
    }

    # 3. 加载轨迹 (Trajectory)
    if not os.path.exists(paths['trajectory']):
        print(f"Trajectory file not found: {paths['trajectory']}")
        return

    trajectories_cache = {}
    with h5py.File(paths['trajectory'], 'r') as f:
        cam_ext_group = f['observation']["camera_extrinsics"]
        trajectories_cache["wrist"] = np.array(cam_ext_group[f"{serials['wrist']}_left"])
        trajectories_cache["ext1"] = np.array(cam_ext_group[f"{serials['ext1']}_left"])
        trajectories_cache["ext2"] = np.array(cam_ext_group[f"{serials['ext2']}_left"])

    # 数据缓存包
    data_cache = {
        'intrinsics': intrinsics_cache,
        'trajectories': trajectories_cache
    }

    # 4. 循环处理三个相机
    # 检查 Video Root 是否存在
    if not os.path.exists(paths['video_root']):
         # 容错：尝试 recordings (有些旧版结构没有 MP4 文件夹)
        fallback_path = os.path.join(raw_seq_path, "recordings")
        if os.path.exists(os.path.join(fallback_path, f"{serials['wrist']}.mp4")):
            paths['video_root'] = fallback_path
        else:
            print(f"Warning: Video directory not found at {paths['video_root']}")

    for cam_name in ["wrist", "ext1", "ext2"]:
        serial = serials[cam_name]
        process_camera_view(cam_name, serial, paths, data_cache, args)

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()