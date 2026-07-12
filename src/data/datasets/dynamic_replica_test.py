import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.dynamic_replica import DynamicReplicaDataset

from src.utils.projection import world_to_pixel_coordinates, closed_form_inverse_se3


class DynamicReplicaTestDataset(BaseTestDataset, DynamicReplicaDataset):
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
        # TODO: handel depth with infinte sky values
        images, foreground_masks, depths, camera_intrinsics = [], [], [], []
        valid_masks = []

        for idx, (image, foreground_mask, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["foreground_mask"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [foreground_mask, depth], camera_intrinsic = self._crop_resize_if_necessary(image, [foreground_mask, depth], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            
            # TODO: check this
            # carefully processs depth
            valid_mask = np.isfinite(depth)
            # depth[~valid_mask] = np.nan
            
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

        # traj: [im h w 3]
        # pts_2d [im h w] pts_2d_valid_mask [im h w] pts_2d_depth [im h w]
        traj_3d_world = scene_data["traj_3d_world"]
        im, n = traj_3d_world.shape[:2]
        h, w = images.shape[1:3]
        t = im # image-per-time
        pts_2d, pts_2d_valid_mask, pts_2d_depth = world_to_pixel_coordinates(traj_3d_world, camera_intrinsics, camera_poses, image_shape=(h, w), return_depth=True, has_batch=True)
        # pts_2d: [im, n, 2], pts_2d_valid_mask: [im, n], depths: [im, h, w]

        # 1. Get integer coordinates for all points
        proj_u = np.rint(pts_2d[..., 0]).astype(np.int32)
        proj_v = np.rint(pts_2d[..., 1]).astype(np.int32)

        # 2. Select indices using valid mask (bounds checked by user)
        b_idx, n_idx = np.where(pts_2d_valid_mask)

        # 3. Extract projected coordinates for valid points
        target_u = proj_u[b_idx, n_idx]
        target_v = proj_v[b_idx, n_idx]

        # 4. Depth Consistency Check
        # Sample GT depth at projected pixels
        gt_depth_val = depths[b_idx, target_v, target_u] 
        # Get calculated depth of the 3D points
        calc_depth_val = pts_2d_depth[b_idx, n_idx]

        # Keep points where calculated depth matches GT depth
        visible_mask = np.abs(calc_depth_val - gt_depth_val) < 0.03

        # 5. Apply visibility filter
        final_b = b_idx[visible_mask]
        final_n = n_idx[visible_mask]
        final_u = target_u[visible_mask]
        final_v = target_v[visible_mask]

        # 6. Fill Trajectory
        trajectory = np.full((im, t, h, w, 3), np.nan, dtype=np.float32)
        point_tracks = traj_3d_world[:, final_n, :]

        trajectory[final_b, :, final_v, final_u] = point_tracks.transpose(1, 0, 2)

        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "dynamic_replica" if dataset_name is None else dataset_name,
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "trajectory": trajectory.astype(np.float32),
            "foreground": foreground_masks.astype(bool),
            "valid_mask": valid_masks.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": True, # 
            "scene_meta": scene_meta,
        }

        return views
