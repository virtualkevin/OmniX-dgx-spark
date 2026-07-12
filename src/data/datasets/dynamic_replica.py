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

class DynamicReplicaDataset(BaseDataset):
    """DynamicReplica dataset, support static_image/unsynced_video/synced_video/hybrid_video
        dataset_info: dict
            num_video: 2
            video_names: ["left", "right"]
        scenes: List[dict]
            scene_info:
                scene_path: str
                num_video_frame: int
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
            image_path = osp.join(scene_path, video_name, "rgb", f"{int(image_name):04d}.png")
            image = imread_cv2(image_path)
            images.append(image)

            # load camera
            camera_path = osp.join(scene_path, video_name, "cam", f"{int(image_name):04d}.npz")
            camera_data = np.load(camera_path)
            camera_intrinsics.append(camera_data["intrinsics"])
            camera_poses.append(camera_data["pose"]) 

            # load foreground_mask
            foreground_mask_path = osp.join(scene_path, video_name, "foreground_mask", f"{int(image_name):04d}.png")
            foreground_mask = imread_cv2(foreground_mask_path)
            foreground_mask = foreground_mask[:, :, 0].astype(np.float32) / 255.0
            foreground_masks.append(foreground_mask)

            # load depth
            depth_path = osp.join(scene_path, video_name, "depth", f"{int(image_name):04d}.npy")
            # check this, far
            depth = np.load(depth_path)
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
        vertex_list = []
        for frame_idx in unique_frame_idxs:
            # load vertex
            # left cam
            vertex_path = osp.join(scene_path, "left", "traj", f"{frame_idx:04d}.npz")
            vertex = np.load(vertex_path)
            vertex_list.append(vertex)
        
        per_timestamp_data = {
            "traj_3d_world": np.stack([vertex["traj_3d_world"] for vertex in vertex_list], axis=0),
        }
            
        return per_timestamp_data

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
            
            # TODO: check this
            # carefully processs depth
            valid_mask = (depth >0) & np.isfinite(depth)
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


