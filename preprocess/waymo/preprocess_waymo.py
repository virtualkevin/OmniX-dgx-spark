import os
import os.path as osp
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.camera_image import CameraName
from waymo_open_dataset.v2.perception.lidar import LaserName, RangeImage, LiDARComponent, LiDARCameraProjectionComponent
from waymo_open_dataset.v2 import LiDARCalibrationComponent

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# -------------------
# 工具函数（保持原有的）
# -------------------

import gc
import psutil, os, ctypes

def trim_memory():
    """尝试将glibc空闲堆内存归还给OS"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def log_memory(stage=""):
    """简单打印当前进程内存占用（MB）"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    print(f"[MEM] {stage}: {mem_mb:.2f} MB", flush=True)

def read_component(dataset_dir, tag, context_name):
    """读取某个 component 的一个 segment parquet"""
    path = osp.join(dataset_dir, tag, f"{context_name}.parquet")
    return pd.read_parquet(path)

def build_K_from_row(row):
    """从 camera_calibration 行构造 3x3 K"""
    fx = row["[CameraCalibrationComponent].intrinsic.f_u"]
    fy = row["[CameraCalibrationComponent].intrinsic.f_v"]
    cx = row["[CameraCalibrationComponent].intrinsic.c_u"]
    cy = row["[CameraCalibrationComponent].intrinsic.c_v"]
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def inv(mat):
    return np.linalg.inv(mat)

def to_hom(X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)]).T

def project_points(K, X_cam):
    Z = np.clip(X_cam[:, 2:3], 1e-6, None)
    uvw = (K @ X_cam.T).T
    u = uvw[:, 0] / Z[:, 0]
    v = uvw[:, 1] / Z[:, 0]
    return u, v, Z[:, 0]

# 车辆系 → OpenCV相机系
AXES_TRANSFORM = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)

DYNAMIC_CLASSES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16]

# -------------------
# 新增：构建映射函数
# -------------------

def build_mappings(timestamps, veh_pose_df, veh_asset_df, lidar_bbox_df, veh_asset_object_ids):
    """一次遍历构建所有映射"""
    # print("Building mappings...")
    
    # 初始化映射字典
    car_pose_map = {}
    frame_objects_map = {}
    object_trajectory_map = {}
    veh_asset_map = {}
    timestamp_to_frame_map = {}
    
    # 构建timestamp到frame索引的映射
    for frame_idx, timestamp in enumerate(timestamps):
        timestamp_to_frame_map[timestamp] = frame_idx
    
    # 构建car_pose_map
    for ts in timestamps:
        if ts in veh_pose_df.index:
            car_pose_row = veh_pose_df.loc[ts]
            car_to_world = np.array(car_pose_row["[VehiclePoseComponent].world_from_vehicle.transform"], dtype=np.float64).reshape(4, 4)
            car_pose_map[ts] = car_to_world
    
    # 构建veh_asset_map
    for ts in timestamps:
        if ts in veh_asset_df.index:
            veh_asset_map[ts] = get_vehicle_assets_for_frame(veh_asset_df, ts)
    
    # 构建物体轨迹映射
    for ts in timestamps:
        if ts in veh_asset_map and veh_asset_map[ts]:
            # 获取当前帧的物体列表
            current_objects = list(veh_asset_map[ts].keys())
            frame_objects_map[ts] = current_objects
            
            # 为每个物体记录轨迹信息
            bbox_dict = get_bboxes_for_frame(ts, veh_asset_object_ids, lidar_bbox_df)
            
            for obj_id in current_objects:
                if obj_id in bbox_dict:
                    frame_idx = timestamp_to_frame_map[ts]
                    
                    if obj_id not in object_trajectory_map:
                        object_trajectory_map[obj_id] = {
                            'timestamps': [],
                            'bboxes': [],
                            'frame_indices': []
                        }
                    
                    object_trajectory_map[obj_id]['timestamps'].append(ts)
                    object_trajectory_map[obj_id]['bboxes'].append(bbox_dict[obj_id])
                    object_trajectory_map[obj_id]['frame_indices'].append(frame_idx)
    
    return {
        'car_pose_map': car_pose_map,
        'frame_objects_map': frame_objects_map,
        'object_trajectory_map': object_trajectory_map,
        'veh_asset_map': veh_asset_map,
        'timestamp_to_frame_map': timestamp_to_frame_map
    }

def compute_vehicle_trajectory(obj_id, current_timestamp, mappings):
    """计算单个物体的世界坐标轨迹"""
    trajectory_info = mappings['object_trajectory_map'][obj_id]
    
    # 获取当前帧的asset点云作为模板
    template_points = mappings['veh_asset_map'][current_timestamp][obj_id]  # [k, 3]
    
    # 计算所有出现帧的世界坐标
    world_trajectory = []
    
    for timestamp, bbox_data in zip(trajectory_info['timestamps'], trajectory_info['bboxes']):
        # 获取car_to_world变换
        car_to_world = mappings['car_pose_map'][timestamp]
        
        # 使用原有的transform_local_pc_to_car函数
        car_points = transform_local_pc_to_car(template_points, bbox_data)
        
        # 转换到世界坐标系
        car_points_hom = np.hstack([car_points, np.ones((car_points.shape[0], 1))])
        world_points = (car_to_world @ car_points_hom.T).T[:, :3]  # [k, 3]
        
        world_trajectory.append(world_points)
    
    world_trajectory = np.stack(world_trajectory, axis=0)  # [t, k, 3]
    
    return {
        'world_trajectory': world_trajectory,
        'frame_indices': trajectory_info['frame_indices']
    }

# -------------------
# 原有的辅助函数（保持不变）
# -------------------

def get_vehicle_assets_for_frame(veh_asset_df, ts):
    """根据时间戳 ts 从 veh_asset_df 获取 {object_id: local_point_cloud} 字典"""
    assets_map = {}
    if ts not in veh_asset_df.index:
        return assets_map
    
    df_frame = veh_asset_df.loc[ts]
    if isinstance(df_frame, pd.Series):
        df_frame = df_frame.to_frame().T

    for _, row in df_frame.iterrows():
        obj_id = row["key.laser_object_id"]
        xyz = row['[ObjectAssetLiDARSensorComponent].points_xyz.values']
        shape = row['[ObjectAssetLiDARSensorComponent].points_xyz.shape']
        points = xyz.reshape(shape)
        assets_map[obj_id] = points

    return assets_map

def get_bboxes_for_frame(ts, veh_asset_object_ids, lidar_bbox_df):
    """根据时间戳 ts 和 object_id 列表，从 lidar_bbox_df 获取 bbox 数据"""
    bbox_cols = [
        "[LiDARBoxComponent].box.center.x",
        "[LiDARBoxComponent].box.center.y",
        "[LiDARBoxComponent].box.center.z",
        "[LiDARBoxComponent].box.size.x",
        "[LiDARBoxComponent].box.size.y",
        "[LiDARBoxComponent].box.size.z",
        "[LiDARBoxComponent].box.heading"
    ]
    
    bbox_dict = {}
    for obj_id in veh_asset_object_ids:
        if (ts, obj_id) in lidar_bbox_df.index:
            row = lidar_bbox_df.loc[(ts, obj_id)]
            if isinstance(row, pd.Series):
                row = row.to_frame().T
            bbox_dict[obj_id] = row[bbox_cols].to_numpy(dtype=np.float64).squeeze()
    
    return bbox_dict

def get_keypoints_for_frame(ts, kp_df):
    """根据时间戳 ts，从 kp_df 提取关键点数据"""
    keypoint_types = [0, 1, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]
    num_types = len(keypoint_types)
    type_to_index = {t: i for i, t in enumerate(keypoint_types)}

    kp_dict = {}
    if ts not in kp_df.index:
        return kp_dict

    df_frame = kp_df.loc[ts]
    if isinstance(df_frame, pd.Series):
        df_frame = df_frame.to_frame().T

    object_ids = df_frame["key.laser_object_id"].unique()

    for obj_id in object_ids:
        df_obj = df_frame[df_frame["key.laser_object_id"] == obj_id]
        if df_obj.empty:
            continue

        kp_array = np.full((num_types, 4), np.nan, dtype=np.float32)

        for _, row in df_obj.iterrows():
            types = np.atleast_1d(row["[LiDARHumanKeypointsComponent].lidar_keypoints[*].type"])
            xs = np.atleast_1d(row["[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.x"])
            ys = np.atleast_1d(row["[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.y"])
            zs = np.atleast_1d(row["[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.z"])
            vis = np.atleast_1d(row["[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.visibility.is_occluded"])

            for k_type_val, x, y, z, v in zip(types, xs, ys, zs, vis):
                k_type = int(k_type_val)
                if k_type not in type_to_index:
                    continue
                idx = type_to_index[k_type]
                kp_array[idx] = np.array([x, y, z, v], dtype=np.float32)

        kp_dict[obj_id] = kp_array

    return kp_dict

def transform_local_pc_to_car(local_pc, bbox):
    """local_pc: Nx3 局部坐标点云, bbox: [cx, cy, cz, length, width, height, heading]"""
    cx, cy, cz, l, w, h, heading = bbox
    rot = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    translation = np.array([cx, cy, cz], dtype=np.float32)
    return (local_pc @ rot.T) + translation

# -------------------
# 主处理函数（重写）
# -------------------

def process_segment(
        dataset_dir, 
        context_name,
        out_root="out_dataset",
        log_root="logs",
        resize_height=512,
    ):
    try:
        # 新增：log文件夹路径（在子进程中也可访问，和主进程一致）
        log_start = os.path.join(log_root, "start")
        log_success = os.path.join(log_root, "success")
        log_error = os.path.join(log_root, "error")
        for d in [log_start, log_success, log_error]:
            os.makedirs(d, exist_ok=True)

        # 【在任务真正开始时写 start 文件】
        with open(os.path.join(log_start, f"{context_name}.txt"), "w") as f:
            f.write("started")

        os.makedirs(out_root, exist_ok=True)
        
        # 创建输出目录
        segment_out = os.path.join(out_root, context_name)
        image_out = os.path.join(segment_out, "image")
        static_mask_out = os.path.join(segment_out, "static_mask")
        pts3d_out = os.path.join(segment_out, "pts3d")
        camera_out = os.path.join(segment_out, "camera")
        veh_trajectory_out = os.path.join(segment_out, "veh_trajectory")
        # lidar_hkp_out = os.path.join(segment_out, "lidar_hkp")
        
        for dir_path in [image_out, static_mask_out, pts3d_out, camera_out, veh_trajectory_out]:
            os.makedirs(dir_path, exist_ok=True)
        
        for dir_path in [image_out, static_mask_out, camera_out]:
            for camera_id in range(1, 6):
                os.makedirs(os.path.join(dir_path, f"cam_{camera_id}"), exist_ok=True)
                
        # 读取数据表
        # print("Loading data components...")
        cam_img_df = read_component(dataset_dir, "camera_image", context_name)
        cam_calib_df = read_component(dataset_dir, "camera_calibration", context_name)
        veh_pose_df = read_component(dataset_dir, "vehicle_pose", context_name)
        lidar_df = read_component(dataset_dir, "lidar", context_name)
        lidar_calib_df = read_component(dataset_dir, "lidar_calibration", context_name)
        veh_asset_df = read_component(dataset_dir, "veh_asset_lidar_sensor", context_name)
        lidar_bbox_df = read_component(dataset_dir, "lidar_box", context_name)
        kp_df = read_component(dataset_dir, "lidar_hkp", context_name)
        cam_seg_df = read_component(dataset_dir, "camera_segmentation", context_name)

        # 建索引
        cam_img_df = cam_img_df.set_index(["key.frame_timestamp_micros", "key.camera_name"])
        cam_seg_df = cam_seg_df.set_index(["key.frame_timestamp_micros", "key.camera_name"])
        veh_pose_df = veh_pose_df.set_index("key.frame_timestamp_micros")
        lidar_df = lidar_df.set_index("key.frame_timestamp_micros")
        veh_asset_df = veh_asset_df.set_index("key.frame_timestamp_micros")
        kp_df = kp_df.set_index("key.frame_timestamp_micros")
        lidar_bbox_df = lidar_bbox_df.set_index(["key.frame_timestamp_micros", "key.laser_object_id"])

        # 获取timestamps和veh_asset_object_ids
        timestamps = sorted(cam_img_df.index.get_level_values(0).unique())
        veh_asset_object_ids = set(veh_asset_df["key.laser_object_id"].unique())

        # 加载相机标定信息
        cam_calib_map = dict()
        for cam_idx, cam_calib_row in cam_calib_df.iterrows():
            cam_name = cam_calib_row["key.camera_name"]
            cam_calib_map[cam_name] = dict()
            cam_calib_map[cam_name]["intrinsic"] = build_K_from_row(cam_calib_row)
            cam_calib_map[cam_name]["cam_to_car"] = np.array(cam_calib_row["[CameraCalibrationComponent].extrinsic.transform"], dtype=np.float64).reshape(4, 4)
            cam_calib_map[cam_name]["image_shape"] = (cam_calib_row["[CameraCalibrationComponent].height"], cam_calib_row["[CameraCalibrationComponent].width"])

        # 加载LiDAR标定信息
        lidar_calib_map = {
            row["key.laser_name"]: LiDARCalibrationComponent.from_dict(row)
            for _, row in lidar_calib_df.iterrows()
        }
        
        # 阶段1：构建映射（新增的优化）
        mappings = build_mappings(timestamps, veh_pose_df, veh_asset_df, lidar_bbox_df, veh_asset_object_ids)
        
        # 阶段2：逐帧处理
        # print("Processing frames...")
        for frame_idx, ts in enumerate(timestamps):
            
            # 1. 处理LiDAR点云（保持原有逻辑）
            if ts in lidar_df.index:
                lidar_rows = lidar_df.loc[ts]
                lidar_pts_list = []
                
                if isinstance(lidar_rows, pd.Series):
                    lidar_rows = lidar_rows.to_frame().T
                    
                for _, lrow in lidar_rows.iterrows():
                    has_pts = lrow["[LiDARComponent].range_image_return1.values"] is not None
                    if has_pts:
                        range_image = RangeImage(
                            values=lrow["[LiDARComponent].range_image_return1.values"],
                            shape=lrow["[LiDARComponent].range_image_return1.shape"]
                        )
                        laser_name = lrow["key.laser_name"]
                        calib_comp = lidar_calib_map[laser_name]

                        pc1 = v2.convert_range_image_to_point_cloud(
                            range_image=range_image,
                            calibration=calib_comp
                        )
                        pc1 = np.asarray(pc1)
                        if pc1.size > 0:
                            lidar_pts_list.append(pc1)

                if lidar_pts_list:
                    lidar_pts = np.concatenate(lidar_pts_list, axis=0).astype(np.float32)
                    
                    # 转换到世界坐标系（新增）
                    if ts in mappings['car_pose_map']:
                        car_to_world = mappings['car_pose_map'][ts]
                        lidar_pts_hom = np.hstack([lidar_pts, np.ones((lidar_pts.shape[0], 1))])
                        world_lidar_pts = (car_to_world @ lidar_pts_hom.T).T[:, :3]
                    else:
                        raise IndexError
                    
                    pts3d_path = os.path.join(pts3d_out, f"frame_{frame_idx:04d}.npz")
                    np.savez_compressed(pts3d_path, points=world_lidar_pts)

            # 2. 处理Vehicle轨迹（新的逻辑）
            if ts in mappings['frame_objects_map']:
                current_objects = mappings['frame_objects_map'][ts]
                trajectories_to_save = {}
                
                for obj_id in current_objects:
                    trajectory_data = compute_vehicle_trajectory(obj_id, ts, mappings)
                    # 扁平化存储，避免嵌套字典
                    trajectories_to_save[f"{obj_id}_trajectory"] = trajectory_data['world_trajectory']
                    trajectories_to_save[f"{obj_id}_frame"] = np.array(trajectory_data['frame_indices'], dtype=np.int32)
                
                # 保存当前帧的轨迹数据
                veh_trajectory_path = os.path.join(veh_trajectory_out, f"frame_{frame_idx:04d}.npz")
                np.savez_compressed(veh_trajectory_path, **trajectories_to_save)

            # # 3. 处理关键点（保持原有逻辑）
            # kp_dict = get_keypoints_for_frame(ts, kp_df)
            # kp_path = os.path.join(lidar_hkp_out, f"{frame_idx}.npz")
            # np.savez_compressed(kp_path, **kp_dict)

            # 4. 处理相机数据（保持原有逻辑，但改进相机参数计算）
            for cam_name in cam_calib_map:
                if (ts, cam_name) in cam_img_df.index:
                    # 处理图像
                    img_row = cam_img_df.loc[(ts, cam_name)]
                    img_bgr = cv2.imdecode(np.frombuffer(img_row["[CameraImageComponent].image"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    scale_factor = resize_height / img_bgr.shape[0]
                    img_bgr = cv2.resize(img_bgr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                    
                    img_path = os.path.join(image_out, f"cam_{cam_name}", f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(img_path, img_bgr)

                    # 处理分割掩码
                    if (ts, cam_name) in cam_seg_df.index:
                        seg_row = cam_seg_df.loc[(ts, cam_name)]
                        divisor = int(seg_row["[CameraSegmentationLabelComponent].panoptic_label_divisor"])
                        panoptic_label = cv2.imdecode(
                            np.frombuffer(seg_row["[CameraSegmentationLabelComponent].panoptic_label"], np.uint8),
                            cv2.IMREAD_UNCHANGED
                        )
                        semantic_map = panoptic_label // divisor
                        static_mask = (~np.isin(semantic_map, DYNAMIC_CLASSES)).astype(np.uint8) * 255
                        static_mask = cv2.resize(static_mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                        
                        static_mask_path = os.path.join(static_mask_out, f"cam_{cam_name}", f"frame_{frame_idx:04d}.png")
                        cv2.imwrite(static_mask_path, static_mask)

                    # 处理相机参数（使用新的变换）
                    cam_intrinsic = cam_calib_map[cam_name]["intrinsic"].copy()
                    cam_intrinsic[0, 0] *= scale_factor
                    cam_intrinsic[1, 1] *= scale_factor
                    cam_intrinsic[0, 2] *= scale_factor
                    cam_intrinsic[1, 2] *= scale_factor
                    
                    cam_to_car = cam_calib_map[cam_name]["cam_to_car"].copy()
                    
                    if ts in mappings['car_pose_map']:
                        car_to_world = mappings['car_pose_map'][ts]
                        cam_to_world = car_to_world @ cam_to_car @ inv(AXES_TRANSFORM)
                    else:
                        cam_to_world = cam_to_car @ inv(AXES_TRANSFORM)  # fallback
                    
                    camera_path = os.path.join(camera_out, f"cam_{cam_name}", f"frame_{frame_idx:04d}.npz")
                    np.savez(camera_path, 
                            intrinsic=cam_intrinsic, 
                            camera_pose=cam_to_world)
        
        with open(os.path.join(log_success, f"{context_name}.txt"), "w") as f:
            f.write("success")
        return ("success", context_name, None)  # ← 添加这行
    
    # print("Processing completed!")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        with open(os.path.join(log_error, f"{context_name}.txt"), "w") as f:
            f.write(error_msg)
        return ("error", context_name, error_msg)  # ← 添加这行
    finally:
        # ======== 内存释放区 START ========
        # 删掉可能占用大量内存的对象（按你的变量名添加）
        for var_name in [
            "cam_img_df", "cam_calib_df", "veh_pose_df", 
            "lidar_df", "lidar_calib_df", "kp_df", "cam_seg_df",
            "mappings", "timestamps", "veh_asset_df", "lidar_bbox_df"
        ]:
            if var_name in locals():
                del locals()[var_name]

        gc.collect()      # 触发Python垃圾回收
        trim_memory()     # 尝试归还堆内存
        log_memory(f"After freeing context {context_name}")
        
# 在文件顶部，process_segment 函数定义之后，添加包装函数
def process_segment_wrapper(args):
    """包装函数，用于解包参数传递给 process_segment"""
    return process_segment(*args)

if __name__ == "__main__":
    import traceback
    
    dataset_dir = "/cos_zw1/share_304064442/hunyuan/yanqinjiang/waymo_v2/waymo_open_dataset_v_2_0_1/validation"
    context_name_file = "/apdcephfs/private_yanqinjiang/project/dream4d/preprocess/waymo/common_files_validation.txt"

    # 获取所有 context_name
    with open(context_name_file, 'r') as f:
        context_names = [line.strip() for line in f.readlines()]

    out_root = "/cos_zw1/share_304064442/hunyuan/yanqinjiang/processed_data/waymo_v2/validation"
    log_dir = "/apdcephfs/private_yanqinjiang/project/dream4d/preprocess/waymo/logs_val"
    # 新增：log文件夹
    log_root = os.path.join("/apdcephfs/private_yanqinjiang/project/dream4d/preprocess/waymo", "logs")
    log_start = os.path.join(log_root, "start")
    log_success = os.path.join(log_root, "success")
    log_error = os.path.join(log_root, "error")
    for d in [log_start, log_success, log_error]:
        os.makedirs(d, exist_ok=True)
    
    success_context_names = os.listdir(log_success)
    success_context_names = [name.split('.')[0] for name in success_context_names]
    context_names = [name for name in context_names if name not in success_context_names]

    print(f"context to process: {len(context_names)}")

    success_list = []
    error_list = []
    error_details = {}

    max_workers = min(16, os.cpu_count())
    print(f"Using {max_workers} processes to speed up conversion...")
    print(f"Total segments to process: {len(context_names)}")

    # 准备参数列表
    args_list = [(dataset_dir, context_name, out_root, log_root) for context_name in context_names]

    # 使用 spawn 确保子进程是全新的（不会继承父进程的内存状态）
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        completed = 0

        # 使用 imap_unordered，哪个任务先完成就先返回
        for result in pool.imap_unordered(process_segment_wrapper, args_list):
            completed += 1
            try:
                status, name, error_msg = result

                if status == "success":
                    print(f"✅ [{completed}/{len(context_names)}] Success: {name}")
                else:
                    print(f"❌ [{completed}/{len(context_names)}] Failed: {name}")
                    print(f"Error:\n{error_msg}\n")

            except Exception:
                error_msg = traceback.format_exc()
                print(f"❌ [{completed}/{len(context_names)}] Exception in worker")
                print(f"Error:\n{error_msg}")