import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseTestDataset
from src.data.datasets.point_odyssey import PointOdysseyDataset

from src.utils.projection import world_to_pixel_coordinates, closed_form_inverse_se3


class PointOdysseyTestDataset(BaseTestDataset, PointOdysseyDataset):
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
        images, depths, camera_intrinsics = [], [], []
        valid_masks = []

        for idx, (image, depth, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["depth"], scene_data["camera_intrinsic"])):
            image, [depth], camera_intrinsic = self._crop_resize_if_necessary(image, [depth], camera_intrinsic, resolution, \
                rng=None, info=path_info[idx])
            
            # TODO: check this
            # carefully processs depth
            valid_mask = np.isfinite(depth)
            # depth[~valid_mask] = np.nan
            
            images.append(image)
            depths.append(depth)

            camera_intrinsics.append(camera_intrinsic)
            valid_masks.append(valid_mask)
                 
        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)
        camera_intrinsics = np.stack(camera_intrinsics, axis=0)
        camera_poses = np.stack(scene_data["camera_pose"], axis=0)
        valid_masks = np.stack(valid_masks, axis=0)

        # traj: [im h w 3]
        # trajs_3d: [im n 3]
        trajs_3d = scene_data["trajs_3d"]
        valids = scene_data["valids"]
        visibs = scene_data["visibs"] 

        # image-per-time
        h, w = images.shape[1:3]
        im, n = valids.shape[:2]
        t = im
        # Fill Trajectory
        trajectory = np.full((im, t, h, w, 3), np.nan, dtype=np.float32)

        pts_2d, pts_2d_valid_mask, _ = world_to_pixel_coordinates(
            trajs_3d, 
            camera_intrinsics, 
            camera_poses, 
            image_shape=(h, w), 
            return_depth=True, 
            has_batch=True
        )


        proj_u = np.rint(pts_2d[..., 0]).astype(np.int32)
        proj_v = np.rint(pts_2d[..., 1]).astype(np.int32)

        final_mask = pts_2d_valid_mask & (valids > 0) & (visibs > 0)


        b_idx, n_idx = np.where(final_mask)

        target_u = proj_u[b_idx, n_idx]
        target_v = proj_v[b_idx, n_idx]


        source_trajs = trajs_3d[:, n_idx, :].transpose(1, 0, 2)


        trajectory[b_idx, :, target_v, target_u, :] = source_trajs


        dataset_name = None
        if self.dataset_info is not None and self.dataset_info.get("dataset_name", None) is not None:
            dataset_name = self.dataset_info["dataset_name"]
        scene_meta = {
            "dataset_name": "point_odyssey" if dataset_name is None else dataset_name,
            "data_sampling_type": data_sampling_type,
            "scene_path": scene_path,
            "path_info": path_info,
        }

        views = {
            "image": images.astype(np.float32),
            "depth": depths.astype(np.float32),
            "trajectory": trajectory.astype(np.float32),
            "valid_mask": valid_masks.astype(bool),
            "camera_pose": camera_poses.astype(np.float32),
            "intrinsic": camera_intrinsics.astype(np.float32),
            "image_info": index_info,
            "merge_traj": True, # 
            "scene_meta": scene_meta,
        }

        return views
