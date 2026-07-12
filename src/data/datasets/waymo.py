import os.path as osp
import os
import numpy as np
import sys

from copy import deepcopy

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.base.base_dataset import BaseDataset
from src.utils.image import imread_cv2

# cam_1: FRONT
# cam_2: FRONT_LEFT
# cam_3: FRONT_RIGHT
# cam_4: SIDE_LEFT
# cam_5: SIDE_RIGHT
data_sampling_video_group = {
    1: [
        ("cam_1",),
        ("cam_2",),
        ("cam_3",),
        ("cam_4",),
        ("cam_5",),
    ],
    2:[
        ("cam_1", "cam_2"),
        ("cam_1", "cam_3"),
        ("cam_1", "cam_4"),
        ("cam_1", "cam_5"),
        ("cam_2", "cam_4"),
        ("cam_3", "cam_5"),
    ],
    3:[
        ("cam_1", "cam_2", "cam_3"),
        ("cam_1", "cam_2", "cam_4"),
        ("cam_1", "cam_3", "cam_5"),
    ],
    4:[
        ("cam_1", "cam_2", "cam_3", "cam_4"),
        ("cam_1", "cam_2", "cam_3", "cam_5"),
        ("cam_1", "cam_2", "cam_4", "cam_5"),
        ("cam_1", "cam_3", "cam_4", "cam_5"),
    ],
    5:[
        ("cam_1", "cam_2", "cam_3", "cam_4", "cam_5"),
    ]
}

class WaymoDataset(BaseDataset):
    """Waymo dataset, support static_image/unsynced_video/synced_video
        dataset_info: dict
            num_video: 5
            video_names: ["cam_1", "cam_2", "cam_3", "cam_4", "cam_5"]
        scenes: List[dict]
            scene_info:
                scene_path: str
                num_video_frame: int
    """

    ## sampling util
    def _sampling_info(self, data_sampling_type, total_num_image, num_image, num_unsynced_video, 
                   num_synced_video, data_sampling_video_group, rng, scene_info):
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
            data_sampling_video_group: Video grouping information for sampling
            rng: Random number generator
            scene_info: Scene information dictionary (modified in-place)
            
        Returns:
            path_info: Array of shape [N, 2] with [video_name, image_name]
            index_info: Array of shape [N, 4] with [image_idx, video_idx, local_time_idx, global_time_idx]
        """
        # Handle video name selection based on sampling type
        if data_sampling_type == DataSamplingType.StaticImage:
            # Sample from all available videos at a random timestamp
            video_names = self.dataset_info["video_names"]
            scene_info["num_image"] = len(video_names)  # "num_image" not in scene_info_key before
            num_video_frame = scene_info["num_video_frame"]
            sampled_timestamp = rng.choice(num_video_frame)
            
            # Generate image names with timestamp
            image_names = [f"{video_name}/{sampled_timestamp:04d}" for video_name in video_names]
            scene_info["image_names"] = image_names
            
        elif data_sampling_type == DataSamplingType.UnsyncedDynamicVideo:
            # Add surrounding camera views to ensure sufficient overlap
            video_name_group = data_sampling_video_group[num_unsynced_video].copy()
            for cam_idx in range(1, 6):
                video_name_group.append(tuple([f"cam_{cam_idx}"] * num_unsynced_video))
            
            video_names = rng.choice(video_name_group)
            scene_info["video_names"] = video_names
            scene_info["num_video"] = len(video_names)
            
        elif data_sampling_type == DataSamplingType.SyncedDynamicVideo:
            # Sample synchronized video group
            video_name_group = data_sampling_video_group[num_synced_video]
            video_names = rng.choice(video_name_group)
            scene_info["video_names"] = video_names
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
        static_masks = []
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

            # load static_mask
            static_mask_path = osp.join(scene_path, "static_mask", video_name, f"frame_{int(image_name):04d}.png")
            if osp.exists(static_mask_path):
                static_mask = imread_cv2(static_mask_path)
                static_mask = static_mask[:, :, 0].astype(np.float32) / 255.0
            else:
                static_mask = np.zeros_like(image)[:, :, 0].astype(np.float32)
            static_masks.append(static_mask)

        per_image_data = {
            "image": images,
            "static_mask": static_masks,
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
        pts3d_list = []
        veh_trajectory_list = []
        for frame_idx in unique_frame_idxs:
            # load pts3d
            pts3d_path = osp.join(scene_path, "pts3d", f"frame_{frame_idx:04d}.npz")
            pts3d = np.load(pts3d_path)["points"]
            pts3d_list.append(pts3d)

            # load and merge vehicle trajectory
            veh_trajectory_path = osp.join(scene_path, "veh_trajectory", f"frame_{frame_idx:04d}.npz")
            if os.path.exists(veh_trajectory_path):
                veh_trajectory_data = np.load(veh_trajectory_path)
                object_ids = np.unique(["_".join(key.split("_")[:-1]) for key in veh_trajectory_data.keys()])
      
                veh_trajectory_dict = {object_id: {"trajectory": veh_trajectory_data[f"{object_id}_trajectory"], "frame_idx": veh_trajectory_data[f"{object_id}_frame"]} \
                        for object_id in object_ids}
                veh_trajectory = self._merge_veh_asset_trajectory(veh_trajectory_dict, unique_frame_idxs)
            else:
                veh_trajectory = np.full(
                    (len(unique_frame_idxs), 0, 3), 
                    np.nan, 
                    dtype=float
                )
            veh_trajectory_list.append(veh_trajectory)
        
        per_timestamp_data = {
            "pts3d": pts3d_list,
            "veh_trajectory": veh_trajectory_list
        }
            
        return per_timestamp_data
    

    ## trajectory util
    def _merge_veh_asset_trajectory(self, veh_asset_trajectory_dict, target_frame_idxs):
        """Merge vehicle asset trajectories into a single trajectory array
        
        Args: 
            veh_asset_trajectory_dict: dict, key is vehicle_id, value is trajectory
                vehicle_id: dict
                    trajectory: [t_, n_, 3] - trajectory points for this vehicle
                    frame_idxs: [t_] - frame indices corresponding to the points
            target_frame_idxs: list/array [t] - target frame indices
            
        Returns:
            merged_veh_asset_trajectory: array [t, total_n, 3] - merged trajectory, 
                                    filled with nan if no trajectory data
        """
        t = len(target_frame_idxs)
        
        if not veh_asset_trajectory_dict:
            return np.full((t, 0, 3), np.nan, dtype=np.float32)
        
        # Pre-calculate point counts and cumulative offsets for each vehicle
        vehicle_info = []
        total_n = 0
        for vehicle_id, veh_data in veh_asset_trajectory_dict.items():
            n_point = veh_data["trajectory"].shape[1]
            vehicle_info.append((vehicle_id, total_n, n_point))
            total_n += n_point
        
        # Pre-allocate result array
        merged_trajectory = np.full((t, total_n, 3), np.nan, dtype=np.float32)
        target_frame_idxs = np.array(target_frame_idxs)
        
        # Batch process all vehicles using vectorized operations
        for vehicle_id, point_offset, n_point in vehicle_info:
            veh_data = veh_asset_trajectory_dict[vehicle_id]
            veh_points = veh_data["trajectory"]  # [t_, n_, 3]
            veh_frame_idxs = np.array(veh_data["frame_idx"])  # [t_]
            
            # Use broadcasting to find matches: [t, t_] boolean matrix
            match_matrix = target_frame_idxs[:, None] == veh_frame_idxs[None, :]  # [t, t_]
            target_has_match = match_matrix.any(axis=1)  # [t]
            
            if target_has_match.any():
                # Get the matched target indices
                matched_target_idxs = np.where(target_has_match)[0]  # [n_matched]
                matched_veh_idxs = match_matrix[matched_target_idxs].argmax(axis=1)  # [n_matched]
                
                merged_trajectory[matched_target_idxs, point_offset:point_offset + n_point, :] = \
                    veh_points[matched_veh_idxs, :, :]
        
        return merged_trajectory

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
        # but waymo_dataset has special constrains when sampling video (surronding view)
        # to make sure there is enough overlaps between videos, we list video_groups as data_sampling_video_group
        # Unified sampling: handle video selection and image sampling
        path_info, index_info = self._sampling_info(
            data_sampling_type, total_num_image, num_image, num_unsynced_video,
            num_synced_video, data_sampling_video_group, rng, scene_info
        )
        
        scene_data = self._load_data(scene_path, path_info, index_info)

        # process per_image_data
        images, static_masks, camera_intrinsics = [], [], []
        for idx, (image, static_mask, camera_intrinsic) in enumerate(zip(scene_data["image"], scene_data["static_mask"], scene_data["camera_intrinsic"])):
            image, [static_mask], camera_intrinsic = self._crop_resize_if_necessary(image, [static_mask], camera_intrinsic, resolution, \
                rng, info=path_info[idx])
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
        # TODO: should we add weigted_mask for trajectory?
        
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
            "dataset_name": "waymo" if dataset_name is None else dataset_name,
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