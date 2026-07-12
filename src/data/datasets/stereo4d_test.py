import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.stereo4d import Stereo4dDataset

from src.utils.projection import world_to_pixel_coordinates, closed_form_inverse_se3


class Stereo4dTestDataset(BaseTestDataset, Stereo4dDataset):
    """Stereo4d test dataset
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
        images, camera_intrinsics = [], []
        for idx, (image, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["camera_intrinsic"])):
            image, [], camera_intrinsic = self._crop_resize_if_necessary(image, [], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            images.append(image)
            camera_intrinsics.append(camera_intrinsic)
                
        images = np.stack(images, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)

        # proj_traj_raw to traj, image-per-time 
        # [t n 3] -> [im t h w 3]
        trajectory_raw = scene_data["trajectory"] # [im/t n 3], np.nan means empty
        im, n = trajectory_raw.shape[:2]
        h, w = images.shape[1:3]
        t = im
        pts_2d, pts_2d_valid_mask, pts_2d_depth = world_to_pixel_coordinates(trajectory_raw, camera_intrinsics, camera_poses, image_shape=(h, w), return_depth=True, has_batch=True)
        # pts_2d: [im, n, 2], pts_2d_valid_mask: [im, n], depths: [im, h, w]
        
        # 1. Get integer coordinates for all points
        proj_u = np.rint(pts_2d[..., 0]).astype(np.int32)
        proj_v = np.rint(pts_2d[..., 1]).astype(np.int32)

        # 2. Select indices using valid mask (bounds checked by user)
        b_idx, n_idx = np.where(pts_2d_valid_mask)

        # 3. Extract projected coordinates for valid points
        target_u = proj_u[b_idx, n_idx]
        target_v = proj_v[b_idx, n_idx]

        # 6. Fill Trajectory
        trajectory = np.full((im, t, h, w, 3), np.nan, dtype=np.float32)
        point_tracks = trajectory_raw[:, n_idx, :]
        trajectory[b_idx, :, target_v, target_u] = point_tracks.transpose(1, 0, 2)

        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "stereo4d" if dataset_name is None else dataset_name,
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": pts_2d_depth.astype(np.float32),
            "trajectory": trajectory.astype(np.float32),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # we merge it already
            "scene_meta": scene_meta,
        }


        return views
