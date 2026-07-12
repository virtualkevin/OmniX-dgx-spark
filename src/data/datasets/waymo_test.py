import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.waymo import WaymoDataset

class WaymoTestDataset(BaseTestDataset, WaymoDataset):
    """Waymo test dataset
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

        images, static_masks, camera_intrinsics = [], [], []
        for idx, (image, static_mask, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["static_mask"], scene_data["camera_intrinsic"])):
            image, [static_mask], camera_intrinsic = self._crop_resize_if_necessary(image, [static_mask], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            images.append(image)
            static_masks.append(static_mask)
            camera_intrinsics.append(camera_intrinsic)
        
        images = np.stack(images, axis=0)
        static_masks = np.stack(static_masks, axis=0).astype(bool)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)
        

        # process per_timestamp_data
        pts3d_all = self._pad_and_stack_arrays(scene_data["pts3d"], axis=0)
        veh_trajectory_all = self._pad_and_stack_arrays(scene_data["veh_trajectory"], axis=1)

        image_shape = (resolution[1], resolution[0])
        frame_to_image_mapping = index_info[:, -1] # global_frame_idx
        trajectory_all_from_pts3d, depthmaps = self._fill_trajectories_from_pts3d(
            pts3d_all, camera_intrinsics, camera_poses, \
            image_shape, frame_to_image_mapping, static_masks=static_masks, \
            return_depthmaps=True
        )

        trajectory_all_from_trajectory = self._fill_trajectories_from_trajectory(
            veh_trajectory_all, camera_intrinsics, camera_poses, \
            image_shape, frame_to_image_mapping
        )

        trajectory_all, trajectory_foreground = self._merge_trajectories([trajectory_all_from_pts3d, trajectory_all_from_trajectory], \
                                                 merge_strategy="overlay")

        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]

        scene_meta = {
            "dataset_name": "waymo_test" if dataset_name is None else dataset_name,
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depthmaps.astype(np.float32),
            "trajectory": trajectory_all.astype(np.float32),
            "trajectory_foreground": trajectory_foreground.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # we merge it already
            "scene_meta": scene_meta,
        }

        return views