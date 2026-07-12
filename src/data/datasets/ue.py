import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseDataset
from src.utils.projection import depthmap_to_world_coordinates, closed_form_inverse_se3

from src.utils.image import imread_cv2

from sklearn.neighbors import NearestNeighbors
import collections

class UEDataset(BaseDataset):
    """UE dataset, support static_image/unsynced_video/synced_video/hybrid_video
        dataset_info: dict
            num_video: 16
            video_names: ["cam_1", "cam_2", ...]
        scenes: List[dict]
            scene_info:
                scene_path: str
                num_video_frame: int
                video_names: ["cam_00", "cam_01", "cam_02", ...]
    """

    ## sampling util
    def _sampling_info(self, data_sampling_type, total_num_image, num_image, num_unsynced_video, 
                   num_synced_video, rng, scene_info):
        """Unified sampling function for Waymo dataset.
        
        This function handles video name selection and efficient image sampling based on 
        the specified data sampling type, considering Waymo's special constraints for 
        surrounding view cameras.
        
        Args:
            data_sampling_type: DataSamplingType
            total_num_image: Total number of images to sample
            num_image: Number of images per sampling unit
            num_unsynced_video: Number of unsynced videos
            num_synced_video: Number of synced videos  
            rng: Random number generator
            scene_info: Scene information dictionary (modified in-place)
            
        Returns:
            path_info: Array of shape [N, 2] with [video_name, image_name]
            index_info: Array of shape [N, 4] with [image_idx, video_idx, local_time_idx, global_time_idx]
        """
        # Handle video name selection based on sampling type
        # TODO: check this, waymo
        if data_sampling_type == DataSamplingType.StaticImage:
            # Sample from all available videos at a random timestamp
            video_names = scene_info["video_names"] if "video_names" in scene_info else self.dataset_info["video_names"] 
            scene_info["num_image"] = len(video_names)  # "num_image" not in scene_info_key before
            num_video_frame = scene_info["num_video_frame"]
            sampled_timestamp = rng.choice(num_video_frame)
            
            # Generate image names with timestamp
            image_names = [f"{video_name}/{sampled_timestamp:04d}" for video_name in video_names]
            scene_info["image_names"] = image_names
        elif data_sampling_type in [DataSamplingType.UnsyncedDynamicVideo, DataSamplingType.SyncedDynamicVideo]:
            if "video_names" not in scene_info:
                scene_info["video_names"] = deepcopy(self.dataset_info["video_names"])
            video_names = scene_info["video_names"]
            scene_info["num_video"] = len(video_names)
        elif data_sampling_type == DataSamplingType.HybridDynamicVideo:
            # Sample from all available videos at a random timestamp
            video_names = scene_info["video_names"] if "video_names" in scene_info else self.dataset_info["video_names"] 
            scene_info["num_image"] = len(video_names)  # "num_image" not in scene_info_key before
            num_video_frame = scene_info["num_video_frame"]
            sampled_timestamp = rng.choice(num_video_frame)
            
            # Generate image names with timestamp
            image_names = [f"{video_name}/{sampled_timestamp:04d}" for video_name in video_names]
            scene_info["image_names"] = image_names

            if "video_names" not in scene_info:
                scene_info["video_names"] = deepcopy(self.dataset_info["video_names"])
            scene_info["num_video"] = len(video_names)
            
        else:
            raise ValueError(f"Not implemented data_sampling_type: {data_sampling_type}")
        
        # Perform efficient image sampling based on selected videos
        path_info, index_info = self._efficient_sample_image(
            data_sampling_type, total_num_image, num_image, num_unsynced_video,
            num_synced_video, rng, scene_info
        )
        
        return path_info, index_info

    ## loading util
    def _load_data(self, scene_path, path_info, index_info):
        """Load all data for the scene.
        
        Args:
            path_info: Array of shape [N, 2] with [video_name, image_name]
            index_info: Array of shape [N, 4] with [image_idx, video_idx, local_time_idx, global_time_idx]
            
        Returns:
            scene_data: Dictionary containing all loaded data
        """
        scene_data = {}        

        # Load per-image data (camera poses, intrinsics, images, etc.)
        per_image_data = self._load_per_image_data(scene_path, path_info, index_info)
        scene_data.update(per_image_data)
        
        # Load per-timestamp data (3D points, trajectories, etc.)
        per_timestamp_data = self._load_per_timestamp_data(scene_path, path_info, index_info)
        scene_data.update(per_timestamp_data)
        
        return scene_data
    

    ## loading util
    def _load_per_image_data(self, scene_path, path_info, index_info):
        """Load data that varies per image (camera-specific data).
        
        Returns data like:
        - camera_poses: [im, 4, 4]
        - camera_intrinsics: [im, 3, 3] 
        - images: [im, h, w, 3]
        - depth_maps: [im, h, w] (if available)
        """
        
        # Load camera poses, intrinsics, images, etc.
        images = []
        foreground_masks = []
        depths = []
        camera_intrinsics, camera_poses = [], []
        for path_info_single, index_info_single in zip(path_info, index_info):
            video_name, image_name = path_info_single
            # load image
            image_path = osp.join(scene_path, "image", video_name, f"frame_{int(image_name):04d}.jpg")
            image = imread_cv2(image_path)
            images.append(image)

            # load camera
            camera_path = osp.join(scene_path, "camera", video_name, f"frame_{int(image_name):04d}.npz")
            camera_data = np.load(camera_path)
            camera_intrinsics.append(camera_data["intrinsic"])
            camera_poses.append(camera_data["camera_pose"]) 

            # load foreground_mask
            foreground_mask_path = osp.join(scene_path, "foreground_mask", video_name, f"frame_{int(image_name):04d}.png")
            foreground_mask = imread_cv2(foreground_mask_path)
            foreground_mask = foreground_mask[:, :, 0].astype(np.float32) / 255.0
            foreground_masks.append(foreground_mask)

            # load depth
            depth_path = osp.join(scene_path, "depth", video_name, f"frame_{int(image_name):04d}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
            else:
                depth_path = osp.join(scene_path, "depth", video_name, f"frame_{int(image_name):04d}.npz")
                depth = np.load(depth_path)["depth"]
            depths.append(depth)

        per_image_data = {
            "image": images,
            "foreground_mask": foreground_masks,
            "depth": depths,
            "camera_intrinsic": camera_intrinsics,
            "camera_pose": camera_poses,
        }
        return per_image_data

    def _load_per_timestamp_data(self, scene_path, path_info, index_info):
        """Load data that varies per timestamp (temporal data).
        
        Returns data like:
        - pts3d: [t, k, 3] for lidar_pts (world coordinate)
        - veh_asset_trajectory: [t, t, p, 3] for veh_trajectory (world coordinate)
        """
        
        # Extract unique timestamps
        # np.unique sort auto
        global_time_idxs, unique_indices = np.unique(index_info[:, -1], return_index=True) # sort
        unique_frame_idxs = path_info[unique_indices][:, 1] # only need image_name (frame_idx), this unique only means no-repeat timestamp
        unique_frame_idxs = [int(frame_idx) for frame_idx in unique_frame_idxs]
        t = len(unique_frame_idxs)

        # Load temporal data for these timestamps
        # face first
        face_path = os.path.join(scene_path, "face.npz")
        faces_dict = np.load(face_path)

        vertex_list = []
        for frame_idx in unique_frame_idxs:
            # load vertex
            vertex_path = osp.join(scene_path, "vertex", f"frame_{frame_idx:04d}.npz")
            vertex = np.load(vertex_path)
            vertex_list.append(vertex)
        

        vertex_mat, face_mat = self.merge_multiple_object(vertex_list, faces_dict)
        per_timestamp_data = {
            "face": face_mat,
            "vertex": vertex_mat
        }
            
        return per_timestamp_data

    # merge multiple object
    def merge_multiple_object(self, vertex_dict_list, faces_dict):
        """
        多帧多物体合并成全局mesh顶点矩阵和faces
        
        Args:
            vertex_dict_list: list[dict], 每个dict是 {actor_name: np.ndarray 顶点坐标}, 多帧
            faces_dict: dict, {actor_name: list[tuple] faces}, faces里的顶点索引是局部的
        Returns:
            vertices_all_frames: np.ndarray, shape (num_frames, total_vertices, 3)
            faces_global: np.ndarray, shape (total_faces, 3)
        """
        num_frames = len(vertex_dict_list)

        # 确认物体顺序（用 faces_dict 的 key 顺序）
        object_names = list(faces_dict.files)

        # 合并 faces（只需要一次）
        faces_all = []
        vertex_offset = 0
        for actor_name in object_names:
            faces_np = np.array(faces_dict[actor_name], dtype=np.int32) + vertex_offset
            faces_all.append(faces_np)
            # 假设顶点数在所有帧一致
            vertex_offset += vertex_dict_list[0][actor_name].shape[0]
        faces_all_object = np.vstack(faces_all)  # (total_faces, 3)

        # 每帧合并顶点并保存
        vertices_all_frames = []
        for frame_idx in range(num_frames):
            vertices_all = []
            for actor_name in object_names:
                vertices_all.append(vertex_dict_list[frame_idx][actor_name])
            vertices_frame = np.vstack(vertices_all)  # (total_vertices, 3)
            vertices_all_frames.append(vertices_frame)
        
        vertices_all_frames = np.stack(vertices_all_frames, axis=0)  # (num_frames, total_vertices, 3)

        return vertices_all_frames, faces_all_object

    ## main func
    def _get_views(self, idx, data_sampling_type, total_num_image, num_image, num_unsynced_video, \
        num_synced_video, resolution, rng):
        """get_views
            views is a dict
                image: [im h w c], not normalized
                depth_map: [optional] [im h w], filled with nan if it is projected sparse depth
                valid_mask: [default true] [im h w], for rgb/depth/trajectory loss
                fg_mask: [optional] [im h w], for segmentation
                trajectory: [optional] [im t h w xyz], t is timesteps, filled with nan if it is sparse trajectory (euqal to pts3d in multiview dataset)
                camera_pose: [im 4 4], c2w
                intrinsic: [im 3 3], note: principal point must be at the center of the image (vggt style)
                image_info: [im 4], (image_idx, video_idx, local_time_idx, global_time_idx)
                merge_traj: bool, whether we should merge depth_map_pts and trajectory
                scene_meta: dict, contains scene_path, sampling info, etc
        """
        scene_info = deepcopy(self.scenes[idx]) # note
        scene_path = osp.join(self.ROOT, scene_info["scene_path"])

        # change/add video_names and num_video_frames here
        # we avoid to use ranodm_sampling here, leave it to _efficient_sample_image
        # Unified sampling: handle video selection and image sampling
        path_info, index_info = self._sampling_info(
            data_sampling_type, total_num_image, num_image, num_unsynced_video,
            num_synced_video, rng, scene_info
        )

        scene_data = self._load_data(scene_path, path_info, index_info)

        # process per_image_data
        # TODO: handel depth with infinte sky values
        images, foreground_masks, depths, camera_intrinsics = [], [], [], []
        valid_masks = []
        for idx, (image, foreground_mask, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["foreground_mask"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [foreground_mask, depth], camera_intrinsic = self._crop_resize_if_necessary(image, [foreground_mask, depth], camera_intrinsic, resolution, \
                rng, info=path_info[idx])
            
            # carefully processs depth
            valid_mask = (depth > 0) & (depth != 65504)
            depth[~valid_mask] = np.nan
            
            images.append(image)
            foreground_masks.append(foreground_mask)
            depths.append(depth)
            camera_intrinsics.append(camera_intrinsic)
            valid_masks.append(valid_mask)
                
        images = np.stack(images, axis=0)
        foreground_masks = np.stack(foreground_masks, axis=0)
        depths = np.stack(depths, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)
        valid_masks = np.stack(valid_masks, axis=0)
        
        # get depth_pts3d first
        pts3d_from_depth = depthmap_to_world_coordinates(depths, camera_intrinsics, camera_poses, has_batch=True) # [im h w 3]
        foreground_masks = foreground_masks.astype(bool)

        # trajectory_all: [im t h w xyz]
        # trajectory_foreground: [im t h w]
        trajectory_all, trajectory_foreground = [], []
        t = len(scene_data["vertex"])
        # loop each image to get dense trajectory map
        for image_idx in range(len(images)):
            vertex_idx = index_info[image_idx, -1]
            foreground_points = pts3d_from_depth[image_idx][foreground_masks[image_idx]]
            # none fg
            if foreground_points.shape[0] == 0:
                foreground_trajectory_per_image = np.empty(
                    (t, 0, 3),
                    dtype=pts3d_from_depth.dtype
                )
            else:
                assignments, min_distances = self.vectorized_assign_points_to_faces(scene_data["vertex"][vertex_idx], scene_data["face"], foreground_points)
                foreground_trajectory_per_image = self.compute_point_trajectory_vectorized(
                    scene_data["vertex"],
                    vertex_idx, # frame_idx
                    scene_data["face"],
                    foreground_points,
                    assignments,
                ) # [t k 3]
            
            # 初始化当前图像的轨迹: [t h w 3]
            traj = np.tile(pts3d_from_depth[image_idx][None], (t, 1, 1, 1))  # [t h w 3]
            
            # 将前景轨迹填充到对应位置
            traj[:, foreground_masks[image_idx]] = foreground_trajectory_per_image  # [t num_fg 3]
            # 前景mask扩展到时间维度
            fg_mask = np.tile(foreground_masks[image_idx][None], (t, 1, 1))  # [t h w]

            trajectory_all.append(traj)
            trajectory_foreground.append(fg_mask)

        trajectory_all = np.stack(trajectory_all, axis=0)  # [im t h w 3]
        trajectory_foreground = np.stack(trajectory_foreground, axis=0)  # [im t h w]

        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "ue" if dataset_name is None else dataset_name,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "trajectory": trajectory_all.astype(np.float32),
            "trajectory_foreground": trajectory_foreground.astype(bool),
            "valid_mask": valid_masks.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # we merge it already
            "scene_meta": scene_meta,
        }

        return views

    def vectorized_assign_points_to_faces(self, vertices, faces, query_points):
        """向量化版本的点到面分配 (彻底修复 Warning 版)"""
        # 找到最近顶点
        if query_points.ndim == 1:
            query_points = query_points[None, :]
        nbrs = NearestNeighbors(n_neighbors=1).fit(vertices)
        _, indices = nbrs.kneighbors(query_points)
        nearest_vertices = indices[:, 0]
        
        # 构建顶点到面的映射
        vertex_face_map = collections.defaultdict(list)
        for face_idx, face in enumerate(faces):
            for vertex_idx in face:
                vertex_face_map[vertex_idx].append(face_idx)
                
        # 获取候选faces
        max_faces = max(len(vertex_face_map[v]) for v in range(len(vertices)))
        candidate_faces = np.zeros((len(query_points), max_faces), dtype=np.int32)
        valid_mask = np.zeros((len(query_points), max_faces), dtype=bool)
        
        for i, vertex_idx in enumerate(nearest_vertices):
            faces_list = vertex_face_map[vertex_idx]
            candidate_faces[i, :len(faces_list)] = faces_list
            valid_mask[i, :len(faces_list)] = True
        
        # 计算距离和内部判断
        face_vertices = vertices[faces[candidate_faces[valid_mask]]]
        points_expanded = np.repeat(query_points[:, None, :], max_faces, axis=1)[valid_mask]
        
        v1 = face_vertices[:, 1] - face_vertices[:, 0]
        v2 = face_vertices[:, 2] - face_vertices[:, 0]
        normals = np.cross(v1, v2)
        
        # 1. 计算模长
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # 2. 标记退化三角形，但【暂时不要】修改距离为Inf，防止后续计算出现NaN
        degenerate_mask = norms.flatten() < 1e-8 
        
        # 3. 安全除法
        safe_norms = np.where(degenerate_mask[:, None], 1.0, norms)
        normals = normals / safe_norms
        
        # 4. 计算距离 (此时即便是坏三角形，距离也是有限数，不会报错)
        distances_to_plane = np.abs(np.sum((points_expanded - face_vertices[:, 0]) * normals, axis=1))
        
        # 【修改点】：这里不要设为 np.inf，让流程继续走完
        # distances_to_plane[degenerate_mask] = np.inf  <-- 删掉这行
        
        projections = points_expanded - (distances_to_plane[:, None] * normals)
        
        # 计算重心坐标
        v0 = face_vertices[:, 1] - face_vertices[:, 0]
        v1 = face_vertices[:, 2] - face_vertices[:, 0]
        v2 = projections - face_vertices[:, 0]
        
        d00 = np.sum(v0 * v0, axis=1)
        d01 = np.sum(v0 * v1, axis=1)
        d11 = np.sum(v1 * v1, axis=1)
        d20 = np.sum(v2 * v0, axis=1)
        d21 = np.sum(v2 * v1, axis=1)
        
        denom = d00 * d11 - d01 * d01
        
        # 保护重心坐标分母
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        eps = 1e-6
        # 此时 u,v,w 都是有限数，比较运算不会报 Warning
        inside_mask = (v >= -eps) & (w >= -eps) & (u >= -eps)
        
        # 5. 【关键】现在再把退化三角形剔除
        inside_mask[degenerate_mask] = False
        
        # 填充结果矩阵
        distances_reshaped = np.full((len(query_points), max_faces), np.inf)
        
        # 【关键】在这里把坏掉的三角形的距离设为 Inf
        # 如果是 valid 的候选面，且是退化面，设为 Inf；否则填入计算出的距离
        final_distances = np.where(degenerate_mask, np.inf, distances_to_plane)
        distances_reshaped[valid_mask] = final_distances
        
        inside_reshaped = np.zeros((len(query_points), max_faces), dtype=bool)
        inside_reshaped[valid_mask] = inside_mask
        
        # 没在三角形内部的，距离惩罚
        distances_reshaped[~inside_reshaped] += 1e6
        
        min_indices = np.argmin(distances_reshaped, axis=1)
        assignments = candidate_faces[np.arange(len(query_points)), min_indices]
        min_distances = distances_reshaped[np.arange(len(query_points)), min_indices]
        
        # 还原显示用的距离 (即便min_distances是inf，减去1e6也还是inf，没问题)
        min_distances = np.minimum(min_distances - 1e6, min_distances)
        
        return assignments, min_distances


    def compute_point_trajectory_vectorized(self, vertices_mat, vertex_idx, faces, foreground_points, assignments):
        """向量化版本的轨迹计算"""
        T, N, _ = vertices_mat.shape
        K = len(foreground_points)
        
        face_vertices = vertices_mat[:, faces[assignments]]
        P0 = foreground_points
        
        dist_sq = np.sum((face_vertices[vertex_idx] - P0[:, None, :]) ** 2, axis=-1)
        weights = 1.0 / (dist_sq + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        trajectories = np.zeros((T, K, 3))
        trajectories[vertex_idx] = P0
        
        for t in range(T):
            if t == vertex_idx:
                continue

            c0 = np.sum(weights[:, :, None] * face_vertices[vertex_idx], axis=1)
            ct = np.sum(weights[:, :, None] * face_vertices[t], axis=1)
            
            q0 = face_vertices[vertex_idx] - c0[:, None, :]
            qt = face_vertices[t] - ct[:, None, :]
            
            S = np.matmul(qt.transpose(0, 2, 1), weights[:, :, None] * q0)
            
            U, s, Vh = np.linalg.svd(S)
            det = np.linalg.det(U @ Vh)
            U[det < 0, :, -1] *= -1
            R = np.matmul(U, Vh)
            
            trajectories[t] = ct + np.matmul(R, (P0 - c0)[:, :, None]).squeeze(-1)
        
        return trajectories



