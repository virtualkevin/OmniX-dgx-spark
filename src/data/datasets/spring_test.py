import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.spring import SpringDataset

from src.utils.projection import depthmap_to_world_coordinates, closed_form_inverse_se3


class SpringTestDataset(BaseTestDataset, SpringDataset):
    """spring test dataset
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
        images, depths, camera_intrinsics = [], [], []
        for idx, (image, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [depth], camera_intrinsic = self._crop_resize_if_necessary(image, [ depth], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            images.append(image)
            depths.append(depth)
            camera_intrinsics.append(camera_intrinsic)
                
        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)

        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "spring" if dataset_name is None else dataset_name,
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": False, # we merge it already
            "scene_meta": scene_meta,
        }

        return views
