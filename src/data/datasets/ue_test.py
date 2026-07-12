import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.ue import UEDataset

from src.utils.projection import depthmap_to_world_coordinates, closed_form_inverse_se3


class UETestDataset(BaseTestDataset, UEDataset):
    """UE test dataset
        scene_info:
            data_sampling_type: str
            scene_path: str
            resolution: int
            video_names: list[str]
            image_names: list[list[str|int]]
            is_synced: bool
    """

    def _get_views(self, idx):
 
        scene_info = deepcopy(self.scenes[idx])
        data_sampling_type = scene_info["data_sampling_type"]

        scene_path = scene_info["scene_path"]
        scene_path = os.path.join(self.ROOT, scene_path)
        video_names = scene_info["video_names"]
        image_names = scene_info["image_names"]
        is_synced = scene_info.get("is_synced", False)

        resolution = scene_info.get("resolution", None)
        if resolution is None:
            resolution = self._resolution

        path_info, index_info = self._get_image_info(
            data_sampling_type, video_names, image_names, is_synced
        )
   
        scene_data = self._load_data(scene_path, path_info, index_info)

        # process per_image_data
        assert self._aug_crop == 1.0, "we do not use aug_crop when evaluation"

        # process per_image_data
        images, foreground_masks, depths, camera_intrinsics = [], [], [], []
        for idx, (image, foreground_mask, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["foreground_mask"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [foreground_mask, depth], camera_intrinsic = self._crop_resize_if_necessary(image, [foreground_mask, depth], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            images.append(image)
            foreground_masks.append(foreground_mask)
            depths.append(depth)
            camera_intrinsics.append(camera_intrinsic)
                
        images = np.stack(images, axis=0)
        foreground_masks = np.stack(foreground_masks, axis=0)
        depths = np.stack(depths, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)

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
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "trajectory": trajectory_all.astype(np.float32),
            "trajectory_foreground": trajectory_foreground.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # we merge it already
            "scene_meta": scene_meta,
        }

        return views
