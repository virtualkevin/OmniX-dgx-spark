import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

import cv2

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseDataset
from src.utils.projection import world_to_pixel_coordinates, closed_form_inverse_se3

from src.utils.image import imread_cv2

from sklearn.neighbors import NearestNeighbors
import collections

class DL3DVDataset(BaseDataset):
    """DL3DV dataset, support static_image
        dataset_info: dict
            num_sample_image: int
            video_names: [""]
        scenes: List[dict]
            scene_info:
                scene_path: str
                num_image: int
    """

    ## sampling util
    def _sampling_info(self, data_sampling_type, total_num_image, num_image, num_unsynced_video, 
                   num_synced_video, rng, scene_info):
        """Unified sampling function for DL3DV dataset.
        
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
            # 1. 明确采样目标数量
            target_num = num_image  
            num_sample_image_pool = self.dataset_info["num_sample_image_pool"] # 采样窗口长度
            scene_num_image = scene_info["num_image"] # 视频总帧数

            # 2. 执行采样逻辑
            if scene_num_image <= num_sample_image_pool:
                # 情况 A: 视频总长不足池子大小
                # 如果总帧数 < 目标采样数，必须设置 replace=True 才能凑够张数
                should_replace = scene_num_image < target_num
                sampled_timestamp = rng.choice(scene_num_image, size=target_num, replace=should_replace)
            else:
                # 情况 B: 视频总长超过池子大小，先确定窗口起点
                start_idx = rng.integers(low=0, high=scene_num_image - num_sample_image_pool + 1)
                
                # 如果池子大小 < 目标采样数，必须设置 replace=True
                should_replace = num_sample_image_pool < target_num
                relative_indices = rng.choice(num_sample_image_pool, size=target_num, replace=should_replace)
                sampled_timestamp = start_idx + relative_indices
            
            # 3. 对时间戳排序（可选，通常视频采样按时间顺序排列更合理）
            sampled_timestamp.sort()

            # 4. 生成图片名称并更新 scene_info
            # 这样生成的 image_names 长度一定是 target_num (即 num_image)
            image_names = [f"/{(timestamp+1):05d}" for timestamp in sampled_timestamp]
            
            scene_info["image_names"] = image_names
            # 直接使用 len 赋值，确保字段与列表长度绝对同步
            scene_info["num_image"] = len(image_names) 
    
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
        
        # # Load per-timestamp data (3D points, trajectories, etc.)
        # per_timestamp_data = self._load_per_timestamp_data(scene_path, path_info, index_info)
        # scene_data.update(per_timestamp_data)

        return scene_data
    

    ## loading util
    # TODO: outlier_mask, consistent with cut3r
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
        depths = []
        camera_intrinsics, camera_poses = [], []
        for path_info_single, index_info_single in zip(path_info, index_info):
            video_name, image_name = path_info_single
            # load image
            image_path = osp.join(scene_path, video_name, "dense", "rgb", f"frame_{int(image_name):05d}.png")
            image = imread_cv2(image_path)
            images.append(image)

            # load camera
            camera_path = osp.join(scene_path, video_name, "dense", "cam", f"frame_{int(image_name):05d}.npz")
            camera_data = np.load(camera_path)
            camera_intrinsics.append(camera_data["intrinsic"])
            camera_poses.append(camera_data["pose"]) 

            # load depth
            depth_path = osp.join(scene_path, video_name, "dense", "depth", f"frame_{int(image_name):05d}.npy")
            depth = np.load(depth_path)

            # load outlier_mask
            sky_mask_path = osp.join(scene_path, video_name, "dense", "sky_mask", f"frame_{int(image_name):05d}.png")
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_UNCHANGED)
            outlier_mask_path = osp.join(scene_path, video_name, "dense", "outlier_mask", f"frame_{int(image_name):05d}.png")
            outlier_mask = cv2.imread(outlier_mask_path, cv2.IMREAD_UNCHANGED)

            depth[sky_mask] = np.nan
            depth[outlier_mask] = np.nan

            depths.append(depth)

        per_image_data = {
            "image": images,
            "depth": depths,
            "camera_intrinsic": camera_intrinsics,
            "camera_pose": camera_poses,
        }
        return per_image_data


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
        images, depths, camera_intrinsics = [], [], []
        valid_masks = []

        for idx, (image, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [depth], camera_intrinsic = self._crop_resize_if_necessary(image, [ depth], camera_intrinsic, resolution, \
                rng, info=path_info[idx])
            
            # TODO: check this
            # carefully processs depth
            valid_mask = (depth > 0) & np.isfinite(depth)
            depth[~valid_mask] = np.nan # for depth loss
            
            images.append(image)
            depths.append(depth)

            camera_intrinsics.append(camera_intrinsic)
            valid_masks.append(valid_mask)
                
        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)
        valid_masks = np.stack(valid_masks, axis=0)


        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "dl3dv" if dataset_name is None else dataset_name,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "valid_mask": valid_masks.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # 
            "scene_meta": scene_meta,
        }

        return views


