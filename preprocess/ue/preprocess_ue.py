import os
import numpy as np
import math
import cv2

from tqdm import tqdm

import json
import OpenEXR
import Imath
import struct

import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime

### read_from_exr
def process_exr_file(file_path, num_camera=16):
    """处理单个EXR文件"""
    try:
        exr_name = os.path.basename(file_path)[:-4]
        exr_file = OpenEXR.InputFile(file_path)

        # 获取图像尺寸
        dw = exr_file.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        available_channels = exr_file.header()['channels'].keys()
        channel_mapping = {}
        needed_channels = []

        for camera_id in range(num_camera):
            # 统一构建 ID 字符串，例如 0 -> "00", 1 -> "01"
            camera_id_str = f"{camera_id:02d}"
            
            if camera_id == 0:
                # Camera 0 RGB 通常是 R, G, B，但也可能带前缀，这里沿用之前的逻辑默认 RGB
                rgb_names = ['R', 'G', 'B']
                # 如果找不到简写，尝试带前缀的（增强健壮性）
                if 'R' not in available_channels and f'FinalImage_CusCamera_00.R' in available_channels:
                     rgb_names = [
                        'FinalImage_CusCamera_00.R',
                        'FinalImage_CusCamera_00.G',
                        'FinalImage_CusCamera_00.B'
                    ]
            else:
                rgb_names = [
                    f'FinalImage_CusCamera_{camera_id_str}.R',
                    f'FinalImage_CusCamera_{camera_id_str}.G',
                    f'FinalImage_CusCamera_{camera_id_str}.B'
                ]

            # 深度通道
            depth_name = f'FinalImageMovieRenderQueue_WorldDepth_CusCamera_{camera_id_str}.R'
            
            # 【修改点】新的 Mask 通道 (0-1值)
            mask_name = f"FinalImageCustomDepthNew_CusCamera_{camera_id_str}.R"

            # 收集 RGB 索引
            rgb_indices = []
            for rgb_name in rgb_names:
                if rgb_name in available_channels:
                    needed_channels.append(rgb_name)
                    rgb_indices.append(len(needed_channels) - 1)

            depth_index = None
            mask_index = None

            # 收集 Depth 索引
            if depth_name in available_channels:
                needed_channels.append(depth_name)
                depth_index = len(needed_channels) - 1

            # 收集 Mask 索引
            if mask_name in available_channels:
                needed_channels.append(mask_name)
                mask_index = len(needed_channels) - 1

            # 如果 RGB(3个), Depth, Mask 都齐全，则记录
            if len(rgb_indices) == 3 and depth_index is not None and mask_index is not None:
                channel_mapping[camera_id] = (rgb_indices, depth_index, mask_index)

        # 批量读取数据
        if not needed_channels:
            return None
            
        channels_data = exr_file.channels(needed_channels, FLOAT)
        exr_file.close()
        
        # 处理每个相机
        results = {}
        for camera_id, (rgb_channels, depth_channel, mask_channel) in channel_mapping.items():
            rgb_img, depth, mask_img = process_exr_file_single_camera(
                file_path,
                camera_id,
                channels_data,
                height,
                width,
                rgb_channels,
                depth_channel,
                mask_channel,
                apply_gamma=False,
            )
            # 返回 RGB(uint8), Depth(float), Mask(uint8)
            results[camera_id] = [rgb_img, depth, mask_img]

        return results

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
    

def process_exr_file_single_camera(file, camera_id, channels_data, height, width,
                        rgb_channels, depth_channel, mask_channel,
                        apply_gamma=True):
    """处理单个相机的RGB、depth和mask数据"""
    try:
        # 1. 处理 RGB
        rgb = np.stack([
            np.frombuffer(channels_data[rgb_channels[0]], dtype=np.float32),
            np.frombuffer(channels_data[rgb_channels[1]], dtype=np.float32),
            np.frombuffer(channels_data[rgb_channels[2]], dtype=np.float32)
        ], axis=-1).reshape(height, width, 3)

        # 2. 处理 Depth
        depth = np.frombuffer(channels_data[depth_channel], dtype=np.float32).reshape(height, width)

        # 3. 处理 Mask (直接读取 New 通道)
        mask = np.frombuffer(channels_data[mask_channel], dtype=np.float32).reshape(height, width)

        # 4. 统一左右翻转
        rgb = cv2.flip(rgb, 1)
        depth = cv2.flip(depth, 1)
        mask = cv2.flip(mask, 1)

        # 5. RGB Gamma 变换
        if apply_gamma:
            rgb = np.clip(rgb, 0, None)
            rgb = np.power(rgb, 1 / 2.2)

        # 6. RGB 转 uint8
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

        # 7. 【修改点】Mask 转 uint8 (0-1 -> 0-255)
        # 假设 mask 是 0-1 的 float，乘以 255 后转 uint8 即可保存为图片
        mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)

        return rgb_uint8, depth, mask_uint8

    except Exception as e:
        print(f"Error processing camera {camera_id} for file {file}: {str(e)}")
        return None, None, None

### process camera_data
def process_camera_data(file_path):
    with open(file_path, "r") as file_to_read:
        camera = json.load(file_to_read)
    
    fov = camera["fov"]
    camera_position = camera["camera_position"]
    camera_rotation = camera["rotation"]

    fov = np.array(fov) # per-camera
    camera_position = np.array(camera_position) # per-camera per-frame
    camera_rotation = np.array(camera_rotation) # per-camera per-frame

    return fov, camera_position, camera_rotation

def compute_camera_intrinsic(hfovs, image_shape):
    """计算相机内参矩阵"""
    hfovs = np.asarray(hfovs, dtype=np.float32)
    h, w = image_shape
    half_hfov = hfovs * 0.5

    fx = w * 0.5 / np.tan(np.radians(half_hfov))
    fy = fx
    cx = w / 2
    cy = h / 2

    K = np.zeros((len(hfovs), 3, 3), dtype=np.float32)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    return K


def compute_camera_c2w_from_pos_rotation(camera_locations, camera_rotations):
    """计算从 UE 坐标到 OpenCV 坐标的 c2w 矩阵"""
    camera_locations = np.asarray(camera_locations, dtype=np.float32)
    camera_rotations = np.asarray(camera_rotations, dtype=np.float32)

    rotations_rad = np.radians(camera_rotations)
    roll = rotations_rad[:, 0]
    pitch = rotations_rad[:, 1]
    yaw = rotations_rad[:, 2]

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    R_yaw = np.stack([
        np.stack([cos_yaw, -sin_yaw, np.zeros_like(yaw)], axis=1),
        np.stack([sin_yaw,  cos_yaw, np.zeros_like(yaw)], axis=1),
        np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)], axis=1)
    ], axis=1)

    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    R_pitch = np.stack([
        np.stack([cos_pitch, np.zeros_like(pitch), -sin_pitch], axis=1),
        np.stack([np.zeros_like(pitch), np.ones_like(pitch), np.zeros_like(pitch)], axis=1),
        np.stack([sin_pitch, np.zeros_like(pitch), cos_pitch], axis=1)
    ], axis=1)

    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    R_roll = np.stack([
        np.stack([np.ones_like(roll), np.zeros_like(roll), np.zeros_like(roll)], axis=1),
        np.stack([np.zeros_like(roll), cos_roll, -sin_roll], axis=1),
        np.stack([np.zeros_like(roll), sin_roll, cos_roll], axis=1)
    ], axis=1)

    R_ue = R_yaw @ R_pitch @ R_roll

    forward_ue = R_ue @ np.array([1, 0, 0])
    right_ue   = R_ue @ np.array([0, 1, 0])
    up_ue      = R_ue @ np.array([0, 0, 1])

    right_cv   = -right_ue
    down_cv    = -up_ue
    forward_cv = forward_ue

    R_opencv = np.stack([right_cv, down_cv, forward_cv], axis=2)

    camera_to_world = np.zeros((len(camera_locations), 4, 4), dtype=np.float32)
    camera_to_world[:, :3, :3] = R_opencv
    camera_to_world[:, :3, 3] = camera_locations
    camera_to_world[:, 3, 3] = 1.0

    return camera_to_world

### read vertices and faces
def read_binary_faces(file_path):
    """读取二进制faces数据文件"""
    faces_dict = {}

    with open(file_path, 'rb') as f:
        header_data = f.read(16)
        magic, version, actor_count, reserved = struct.unpack('<IIII', header_data)
        
        if magic != 0x45434146:  # "FACE"
            raise ValueError(f"Invalid faces file format: magic number is {hex(magic)}")
            
        for _ in range(actor_count):
            actor_header = f.read(8)
            name_length, section_count = struct.unpack('<II', actor_header)
            
            actor_name = f.read(name_length).decode('utf-8')
            
            faces = []
            
            for section_idx in range(section_count):
                section_header = f.read(8)
                section_index, face_count = struct.unpack('<II', section_header)
                
                for _ in range(face_count):
                    face_indices = struct.unpack('<III', f.read(12))
                    faces.append(face_indices)
            
            faces_dict[actor_name] = np.array(faces, dtype=np.int32)

    return faces_dict

def read_binary_vertex_file(file_path):
    """读取二进制顶点数据文件"""
    vertices_dict = {}
    
    with open(file_path, 'rb') as f:
        header_data = f.read(16)
        magic, version, total_frames, reserved = struct.unpack('<IIII', header_data)
        
        if magic != 0x56545844:  # "VTXD"
            raise ValueError(f"Invalid vertex file format: magic number is {hex(magic)}")
            
        frame_data = f.read(16)
        frame_number, actor_count, data_size = struct.unpack('<IIQ', frame_data)
        
        for _ in range(actor_count):
            actor_header_data = f.read(8)
            name_length, vertex_count = struct.unpack('<II', actor_header_data)
            
            actor_name = f.read(name_length).decode('utf-8')
            
            vertices = []
            for _ in range(vertex_count):
                vertex_data = f.read(24)
                x, y, z = struct.unpack('<ddd', vertex_data)
                vertices.append([x, y, z])
            
            vertices_dict[actor_name] = np.array(vertices)
    
    return vertices_dict

### collect and process data
def process_sequence(sequence_path, output_path, image_shape):

    output_cam_folder = os.path.join(output_path, "camera")
    output_image_folder = os.path.join(output_path, "image")
    output_depth_folder = os.path.join(output_path, "depth")
    output_foreground_mask_folder = os.path.join(output_path, "foreground_mask")
    output_vertex_folder = os.path.join(output_path, "vertex")
    output_face_path = os.path.join(output_path, "face.npz")

    os.makedirs(output_cam_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_foreground_mask_folder, exist_ok=True)
    os.makedirs(output_vertex_folder, exist_ok=True)

    # read camera_data
    camera_data_path = os.path.join(sequence_path, os.path.basename(sequence_path).lower() + ".json")
    fov, camera_position, camera_rotation = process_camera_data(camera_data_path)
    num_camera, num_frame = camera_rotation.shape[:2]
    camera_intrinsic = compute_camera_intrinsic(fov, image_shape) 
    camera_c2w = compute_camera_c2w_from_pos_rotation(camera_position.reshape(-1, 3), camera_rotation.reshape(-1, 3))
    camera_c2w = camera_c2w.reshape(num_camera, num_frame, 4, 4)

    for camera_id in range(num_camera):
        os.makedirs(os.path.join(output_cam_folder, f"cam_{camera_id:02d}"), exist_ok=True)
        os.makedirs(os.path.join(output_image_folder, f"cam_{camera_id:02d}"), exist_ok=True)
        os.makedirs(os.path.join(output_depth_folder, f"cam_{camera_id:02d}"), exist_ok=True)
        os.makedirs(os.path.join(output_foreground_mask_folder, f"cam_{camera_id:02d}"), exist_ok=True)

    # save camera_data
    for camera_id in range(num_camera):
        camera_intrinsic_single = camera_intrinsic[camera_id]
        for frame_id in range(num_frame):
            camera_pose = camera_c2w[camera_id, frame_id]
            output_cam_path = os.path.join(output_cam_folder, f"cam_{camera_id:02d}", f"frame_{frame_id:04d}.npz")
            np.savez_compressed(output_cam_path, intrinsic=camera_intrinsic_single, camera_pose=camera_pose)

    # read_geometry_file
    vertices_file_folder = os.path.join(sequence_path, "vertex_data")
    vertices_file_names = os.listdir(vertices_file_folder)
    vertices_file_names.sort() 
    assert num_frame == len(vertices_file_names), f"num_frame {num_frame} != num_geometry_file {len(vertices_file_names)}"
    
    faces_file_path = os.path.join(sequence_path, "faces.bin")
    faces_dict = read_binary_faces(faces_file_path)
    np.savez_compressed(output_face_path, **faces_dict)
    
    for frame_id in range(num_frame):
        vertices_file_path = os.path.join(vertices_file_folder, vertices_file_names[frame_id])
        output_vertex_path = os.path.join(output_vertex_folder, f"frame_{frame_id:04d}.npz")
        vertices_dict = read_binary_vertex_file(vertices_file_path)
        np.savez_compressed(output_vertex_path, **vertices_dict)

    # read exr_file
    exr_file_folder = os.path.join(sequence_path, "images")
    exr_file_names = [f for f in os.listdir(exr_file_folder) if f.endswith('.exr')]
    exr_file_names.sort() # TODO: check this
    exr_file_names = exr_file_names[4:] # TODO: replace this
    assert num_frame == len(exr_file_names), f"num_frame {num_frame} != num_exr_file {len(exr_file_names)}" # tmp, for background
    

    for frame_id in range(num_frame):
        if frame_id >= len(exr_file_names):
            break
            
        exr_file_name = exr_file_names[frame_id]
        exr_file_path = os.path.join(exr_file_folder, exr_file_name)
        
        # 解析 EXR
        camera_data_single_frame = process_exr_file(exr_file_path, num_camera=num_camera)
        
        if camera_data_single_frame is None:
            continue

        for camera_id in range(num_camera):
            if camera_id not in camera_data_single_frame:
                continue

            # 获取处理好的数据
            rgb = camera_data_single_frame[camera_id][0]
            depth = camera_data_single_frame[camera_id][1]
            foreground_mask = camera_data_single_frame[camera_id][2]

            # 保存 RGB
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            output_rgb_path = os.path.join(output_image_folder, f"cam_{camera_id:02d}", f"frame_{exr_file_name[:-4]}.jpg")
            cv2.imwrite(output_rgb_path, bgr)

            # 保存 Depth (cm)
            output_depth_path = os.path.join(output_depth_folder, f"cam_{camera_id:02d}", f"frame_{exr_file_name[:-4]}.npy")
            np.save(output_depth_path, depth)

            # 保存 Foreground Mask
            # 已经是 uint8 (0-255) 图像格式，直接保存
            output_foreground_mask_path = os.path.join(output_foreground_mask_folder, f"cam_{camera_id:02d}", f"frame_{exr_file_name[:-4]}.png")
            cv2.imwrite(output_foreground_mask_path, foreground_mask)

    return

# 设置日志
def setup_logger(output_base, file_map_name):
    """设置日志"""
    log_dir = os.path.join(output_base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{file_map_name}.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger

# 修改后的 process_sequence，添加日志和错误处理
def process_sequence_safe(sequence_path, output_base, image_shape):
    """带错误处理的序列处理函数"""
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(sequence_path):
            logger.warning(f"Sequence path does not exist: {sequence_path}")
            return False
        
        vertex_folder = os.path.join(sequence_path, "vertex_data")
        image_folder = os.path.join(sequence_path, "images")
        faces_file = os.path.join(sequence_path, "faces.bin")
        
        missing_items = []
        if not os.path.exists(vertex_folder):
            missing_items.append("vertex_data")
        if not os.path.exists(image_folder):
            missing_items.append("images")
        if not os.path.exists(faces_file):
            missing_items.append("faces.bin")
            
        if missing_items:
            logger.warning(f"Missing items in {os.path.basename(sequence_path)}: {', '.join(missing_items)}")
            return False
        
        vertex_files = os.listdir(vertex_folder)
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.exr')]
        
        if len(vertex_files) == 0:
            logger.warning(f"Empty vertex folder: {os.path.basename(sequence_path)}")
            return False
            
        if len(image_files) == 0:
            logger.warning(f"Empty image folder: {os.path.basename(sequence_path)}")
            return False
        
        sequence_name = os.path.basename(sequence_path)
        file_map_name = os.path.basename(os.path.dirname(sequence_path))
        
        logger.info(f"Processing {sequence_name}: {len(vertex_files)} vertex files, {len(image_files)} image files")
        
        output_dir = os.path.join(output_base, file_map_name, sequence_name)
        
        process_sequence(sequence_path, output_dir, image_shape)
        
        logger.info(f"✓ Successfully processed: {sequence_name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing {os.path.basename(sequence_path)}: {str(e)}", exc_info=True)
        return False

# 收集单个 FILE_MAP 下的所有 SEQUENCE
def collect_sequences_from_filemap(file_map_path):
    sequences = []
    
    if not os.path.exists(file_map_path):
        print(f"File map path does not exist: {file_map_path}")
        return sequences
    
    if not os.path.isdir(file_map_path):
        print(f"Not a directory: {file_map_path}")
        return sequences
    
    for item in os.listdir(file_map_path):
        item_path = os.path.join(file_map_path, item)
        if os.path.isdir(item_path) and item.startswith("SEQUENCE_"):
            sequences.append(item_path)
    
    sequences.sort()
    return sequences

# 多进程处理单个 FILE_MAP 的所有序列
def process_filemap_parallel(file_map_path, output_base, image_shape, num_workers=None):
    file_map_name = os.path.basename(file_map_path)
    
    logger = setup_logger(output_base, file_map_name)
    logger.info("="*60)
    logger.info(f"Starting to process FILE_MAP: {file_map_name}")
    logger.info(f"File map path: {file_map_path}")
    logger.info(f"Output base: {output_base}")
    logger.info("="*60)
    
    logger.info("Collecting sequences...")
    sequences = collect_sequences_from_filemap(file_map_path) # tmp_debug
    logger.info(f"Found {len(sequences)} sequences")
    
    if len(sequences) == 0:
        logger.warning("No sequences found!")
        return
    
    for i, seq in enumerate(sequences, 1):
        logger.info(f"  {i}. {os.path.basename(seq)}")
    
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    logger.info(f"Using {num_workers} worker processes")
    logger.info("="*60)
    
    process_func = partial(
        process_sequence_safe,
        output_base=output_base,
        image_shape=image_shape
    )
    
    success_count = 0
    failed_count = 0
    failed_sequences = []
    
    with mp.Pool(processes=num_workers) as pool:
        results = []
        for seq_path in sequences:
            result = pool.apply_async(process_func, args=(seq_path,))
            results.append((seq_path, result))
        
        for seq_path, result in tqdm(results, desc=f"Processing {file_map_name}"):
            seq_name = os.path.basename(seq_path)
            try:
                success = result.get(timeout=3600)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_sequences.append(seq_name)
            except mp.TimeoutError:
                logger.error(f"Timeout processing: {seq_name}")
                failed_count += 1
                failed_sequences.append(seq_name)
            except Exception as e:
                logger.error(f"Exception processing {seq_name}: {str(e)}")
                failed_count += 1
                failed_sequences.append(seq_name)
    
    logger.info("="*60)
    logger.info(f"Processing completed for {file_map_name}!")
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"✓ Success: {success_count}")
    logger.info(f"✗ Failed: {failed_count}")
    
    if failed_sequences:
        logger.info("Failed sequences:")
        for seq_name in failed_sequences:
            logger.info(f"  - {seq_name}")
    
    logger.info("="*60)

# 主程序入口
if __name__ == "__main__":
    # 配置你的路径
    file_map_path = "/cos_nj1/share_302245012/hunyuan/yanqinjiang/render_output/20260118_new/Courtroom_Level_jx_08_RUN_1"
    output_base = "/apdcephfs_jn2/share_303535725/yanqinjiang/ue/codes/ue_test_20260118/preprocess/"
    image_shape = (540, 960)
    
    num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    process_filemap_parallel(
        file_map_path=file_map_path,
        output_base=output_base,
        image_shape=image_shape,
        num_workers=num_workers
    )