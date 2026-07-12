import os
import json
import itertools
from collections import defaultdict
import random

import PIL
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.data.datasets.utils.augmentation import ImageAugmentation
import src.data.datasets.utils.cropping as cropping
from src.utils.projection import world_to_camera_coordinates, world_to_pixel_coordinates, depthmap_to_world_coordinates, closed_form_inverse_se3
from src.utils.image import to_tensor

class BaseDataset(Dataset):
    """Define all basic options.
    Args:
        split: str, train or test
        resolution: int or tuple, square_size or (width, height)
        aug_crop: bool, whether to crop and resize image
        normalize_camera_pose: bool, whether to normalize camera pose
        normalize_point_cloud: bool, whether to normalize point cloud
        aug_image: bool, whether to augment image
        seed: int, random seed
        patch_size: int, patch size for image
        transform: ImageAugmentation, image augmentation

        dataset_info: dict, universal info for all scenes, used when scene_info is None
            dataset_name: [optional] str
            num_image: [optional] int, number of images
            num_video: [optional] int, number of videos
            video_names: [optional] list[str], names of video
            num_video_frame: [optional] int, number of frames in each video
            num_oribit_video: [optional] int, number of oribit videos
            num_oribit_video_frame: [optional] int, number of frames in each oribit video

        data_sampling_info: dict, data sampling info for all data sampling types
            data_sampling_type: dict, sampling info for this data sampling type
                data_sampling_ratio: float
                max_total_image: int
                max_video: int
                num_frame_interval_range: [optional], tuple[int, int]
                num_clip_interval_range: [optional], tuple[int, int]
        scene_info_file: str, scene info filename, json file with scenes
            scenes: List[dict], scene_info
                scene_info:
                    scene_name: str
                    scene_path: str
                    num_image: [optional], int
                    image_names: [optional], list[str]
                    sync_video: [optional], bool
                    num_video: [optional], int
                    video_names: [optional], list[str]
                    num_video_frame: [optional], int | list[int]
                    num_oribit_video: [optional], int
                    num_oribit_video_frame: [optional], int | list[int]

    Usage:
        class MyDataset (BaseDataset):
            dataset_info: 
            data_sampling_info: 
            def _get_views(self, idx, *args):
                # get views
                # views is a dict
                #   image: [im h w c] no normalize
                #   depth_map: [optional] [im h w], filled with nan if it is projected sparse depth
                #   valid_mask: [default true] [im h w], for rgb/depth/trajectory loss
                #   fg_mask: [optional] [im h w], for segmentation
                #   trajectory: [optional] [im t h w xyz], t is timesteps, filled with nan if it is sparse trajectory (euqal to pts3d in multiview dataset)
                #   camera_pose: [im 4 4], c2w
                #   intrinsics: [im 3 3], note: principal point must be at the center of the image (vggt style)
                #   image_info: [im 4], (image_idx, video_idx, local_time_idx, global_time_idx)
                #   merge_traj: bool, whether we should merge depth_map_pts and trajectory
                #   scene_meta: dict, contains sampling info, _rgn_seed, scene_name, data_path, etc

                # overload here
                scene_info = self.scenes[idx]
                scene_info[...] = [...] # add more info, especially for hybrid_video_sampling
                path_info, index_info = self._efficient_sample_image(*args)

                views = []
                # load image here
                return views
    """
    def __init__(
        self,
        ROOT=None,
        split=None,
        resolution=None,  # square_size or (width, height)
        aug_crop=False,
        seq_aug_crop=False,
        normalize_camera_pose=False,
        normalize_point_cloud=False,
        aug_image=False,
        seed=None,
        patch_size=14,
        transform=ImageAugmentation(apply_aug=False),
        dataset_info=None, 
        data_sampling_info=None,
        scene_info_file="scene_info",
    ):
        self.ROOT = ROOT
        self.split = split
        self._set_resolution(resolution)
        # aug_crop used in crop_resize_if_necessary, we set it as 1.0 because of maybe depth projection error
        self._aug_crop = aug_crop
        self._seq_aug_crop = seq_aug_crop 
        self._normalize_camera_pose = normalize_camera_pose
        self._normalize_point_cloud = normalize_point_cloud
        self._aug_image = aug_image

        self.seed = seed
        self.patch_size = patch_size
        self.transform = transform

        if dataset_info is not None:
            self.dataset_info = dataset_info
        if data_sampling_info is not None:
            self.data_sampling_info = data_sampling_info
        self.scene_info_file = scene_info_file

        # load_scene_info
        self._load_scene_info()

    def __len__(self):
        return len(self.scenes)
    
    def _load_scene_info(self,):

        if self.split.startswith("/"):
            scene_info_path = f"{self.split}" if self.split.endswith(".json") else f"{self.split}.json"
        else:
            scene_info_path = os.path.join(self.ROOT, f'{self.scene_info_file}_{self.split}.json')
        with open(scene_info_path, 'r') as file_to_read:
            scenes = json.load(file_to_read)
        
        self.scenes = scenes

    ## data processing utils
    def _normalize_by_first_camera_pose(self, views):
        """Normalize by first camera"""
        first_view_camera_pose = views["camera_pose"][0:1]
        inv_first_view_camera_pose = closed_form_inverse_se3(first_view_camera_pose)
        views["camera_pose"] = inv_first_view_camera_pose @ views["camera_pose"]
        if "trajectory" in views:
            views["trajectory"] = world_to_camera_coordinates(views["trajectory"], first_view_camera_pose[0], has_batch=False)

        return views
    
    def _normalized_by_pointcloud_scale(self, views):
        """Normalize by pointcloud average scale"""
        # TODO: should we use valid_mask?
        image_indices = views["image_info"][..., 0]
        global_time_indices = views["image_info"][:, -1]
        pts3d = views["trajectory"][image_indices, global_time_indices]

        pts3d_mask = ~np.isnan(pts3d).any(axis=-1)
        valid_pts3d = pts3d[pts3d_mask]
        # TODO: should be handle this? sometimes it does not have valid points
        # if pts3d_mask.sum() == 0:
        #     with open("/apdcephfs/private_yanqinjiang/project/dream4d/tmp/bad_case.txt", "w+") as file_to_write:
        #         file_to_write.write("image_shape: ")
        #         file_to_write.write(f"{views['image'].shape}\n")
        #         file_to_write.write("image_info: ")
        #         file_to_write.write(f"{views['image_info']}\n")
        #         file_to_write.write("scene_meta: ")
        #         file_to_write.write(f"{views['scene_meta']}\n")

        # TODO: handle all nan, and record bad case
        dist = np.linalg.norm(valid_pts3d, axis=-1)
        norm_factor = np.mean(dist)
        norm_factor = np.nan_to_num(norm_factor, nan=1.0).clip(min=1e-8)

        views["trajectory"] /= norm_factor
        views["depth"] /= norm_factor
        views["camera_pose"][..., :3, 3] /= norm_factor
        views["norm_factor"] = norm_factor
        
        return views
    
    def _process_trajectory(self, views):
        if "depth" in views:
            if ("trajectory" not in views) or ("trajectory" in views and views["merge_traj"]):
                pts3d_from_depth = depthmap_to_world_coordinates(views["depth"], views["intrinsic"], views["camera_pose"], has_batch=True)

                im, h, w = views["image"].shape[:3]
                t = len(np.unique(views["image_info"][:, -1]))
                new_trajectory = np.full((im, t, h, w, 3), np.nan)
    
                image_index = views["image_info"][..., 0]
                global_time_index = views["image_info"][:, -1]
                new_trajectory[image_index, global_time_index] = pts3d_from_depth

                if "trajectory" in views and views["merge_traj"]:
                    trajectory_mask = ~np.isnan(views["trajectory"]).any(axis=-1)
                    new_trajectory[trajectory_mask] = views["trajectory"][trajectory_mask]
            
                views["trajectory"] = new_trajectory.astype(np.float32)

            # for video
            if ("foreground" in views) and ("trajectory_foreground" not in views):
                trajectory_foreground = views["foreground"][:, None].repeat(t, axis=1)
                views["trajectory_foreground"] = trajectory_foreground

        # note: traj_to_depth is not used here, because to get trajectory map, we need to project
        # it to camera, and during this, we will record depth_from_traj

        return views
            

    def _aug_all_view_image(self, views):

        aug_allview_images = self.transform(views["image"])
        views["image"] = aug_allview_images
        
        return views
    
    def _crop_resize_if_necessary(
        self, image, auxiliary_maps, intrinsics, resolution, rng=None, info=None
    ):
        # TODO: change this to include auxiliary_maps (depth_maks, fg_mask, valid_mask, etc)
        
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, auxiliary_maps, intrinsics = cropping.crop_image_auxiliary_maps(
            image, auxiliary_maps, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self._aug_crop > 1:
            target_resolution += (
                rng.integers(0, self._aug_crop)
                if not self._seq_aug_crop
                else self.delta_target_resolution
            )
        elif 0 < self._aug_crop < 1:
            delta_target_ratio = rng.random() * (1. / self.aug_crop - 1.)
            delta_target_resolution = (np.array(resolution) * delta_target_ratio).astype("int")
            target_resolution += (
                delta_target_resolution
                if not self._seq_aug_crop
                else self.delta_target_resolution
            )
            # target_resolution += (
            #     self._rng.integers(0, int(max(resolution) * (1. / self.aug_crop - 1.)))
            #     if not self.seq_aug_crop
            #     else self.delta_target_resolution
            # )
        image, auxiliary_maps, intrinsics = cropping.rescale_image_auxiliary_maps(
            image, auxiliary_maps, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, auxiliary_maps, intrinsics2 = cropping.crop_image_auxiliary_maps(
            image, auxiliary_maps, intrinsics, crop_bbox
        )

        return image, auxiliary_maps, intrinsics2
    
    ## sampling utils
    def _sample_static_image(self, total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info):
        """
            sample static image
            Returns:
                path_info: array, [N,2], last dim [video_name, image_name]
                index_info: array, [N, 3], last dim [video_idx, local_time_idx, global_time_idx], time_idx is defined within video
        """
        scene_num_image = scene_info.get("num_image", None)
        scene_image_names = scene_info.get("image_names", None)
        if scene_num_image is None:
            scene_num_image = self.dataset_info.get("num_image", None)
        assert scene_num_image is not None, "data_sampling for static image must provide num_image in scene_info/dataset_info"
        
        assert scene_image_names is not None and len(scene_image_names)==scene_num_image, \
            "data_sampling for static image must provide image_names in scene_info, \
                and the length must be equal to num_image"
        
        num_sample = num_image if scene_num_image >= num_image else scene_num_image
        sampled_image_names = rng.choice(scene_image_names, num_sample, replace=False)

        num_repeat_image = num_image - num_sample
        
        # by default, we repeat the last image
        sampled_image_names = np.concatenate([sampled_image_names, np.repeat(sampled_image_names[-1], num_repeat_image)])
        sampled_video_names = [sampled_image_name.split("/")[0] for sampled_image_name in sampled_image_names]
        sampled_image_names = [sampled_image_name.split("/")[1] for sampled_image_name in sampled_image_names]
        path_info = np.stack([sampled_video_names, sampled_image_names], axis=1)
        # tmp_debug: modify
        index_info = np.stack([np.arange(num_image).astype(np.int64), np.zeros(num_image).astype(np.int64), np.arange(num_image).astype(np.int64)], axis=1)

        return path_info, index_info

    def _sample_unsynced_video(self, total_num_image, num_image, num_unsynced_video, \
        num_synced_video, rng, scene_info):
        """
            sample unsynced dynamic video
            Returns:
                path_info: array, [N,2], last dim [video_name, image_name]
                index_info: array, [N, 3], last dim [video_idx, local_time_idx, global_time_idx],
        """
        # TODO: by default each video has same time start
        
        # video sampling
        scene_num_video = scene_info.get("num_video", None)
        scene_video_names = scene_info.get("video_names", None)
        if scene_num_video is None:
            scene_num_video = self.dataset_info.get("num_video", None)
        assert scene_num_video is not None, "data_sampling for unsynced dynamic video must provide num_video in scene_info/dataset_info"
        
        assert scene_video_names is not None and len(scene_video_names)==scene_num_video, \
            "data_sampling for unsynced dynamic video must provide video_names in scene_info, \
                and the length must be equal to num_video"

        num_sampled_video = num_unsynced_video if scene_num_video >= num_unsynced_video else scene_num_video
        sampled_video_names_ = rng.choice(scene_video_names, num_sampled_video, replace=False)

        # frame sampling
        num_video_frame = scene_info.get("num_video_frame", None)
        if num_video_frame is None:
            num_video_frame = self.dataset_info.get("num_video_frame", None)
        assert num_video_frame is not None, "data_sampling for unsynced dynamic video must provide num_video_frame in scene_info/dataset_info"
        
        num_frame_interval_range = self.data_sampling_info[DataSamplingType.UnsyncedDynamicVideo].get("num_frame_interval_range", (1,2))
        num_clip_interval_range = self.data_sampling_info[DataSamplingType.HybridDynamicVideo].get("num_clip_interval_range", (3,3))

        num_frame_interval = rng.integers(num_frame_interval_range[0], num_frame_interval_range[1]+1)   
        num_clip_interval = rng.integers(num_clip_interval_range[0], num_clip_interval_range[1]+1)

        # TODO: test this
        min_per_clip = max((total_num_image // num_sampled_video) // 2, 1) # hard-coding
        remainder = total_num_image - (num_sampled_video * min_per_clip)
        cut_points = np.sort(rng.integers(0, remainder + 1, size=num_sampled_video - 1))

        padded_cuts = np.concatenate(([0], cut_points, [remainder]))
        random_parts = np.diff(padded_cuts)
   
        num_clip_frame_list = (random_parts + min_per_clip).tolist()

        frame_idxs_list = self._sample_frame_idx_from_video(num_video_frame, num_clip_frame_list, num_frame_interval, num_clip_interval, rng)
        sampled_video_names = []
        sampled_image_names = []
        video_idxs = []
        local_time_idxs = []
        
        for video_idx, (video_name, frame_idxs) in enumerate(zip(sampled_video_names_, frame_idxs_list)):
            sampled_video_names = sampled_video_names + [video_name] * len(frame_idxs)
            sampled_image_names = sampled_image_names + [f"{frame_idx}" for frame_idx in frame_idxs]
            video_idxs = video_idxs + [video_idx] * len(frame_idxs)
            local_time_idxs = local_time_idxs + [time_idx for time_idx in range(len(frame_idxs))]
        
        global_time_idxs = [time_idx for time_idx in range(len(local_time_idxs))]
            
  
        # if not enough, repeat the last image of the last clip
        num_repeat_image = total_num_image - sum([len(frame_idxs) for frame_idxs in frame_idxs_list])
        if num_repeat_image > 0:
            sampled_video_names = sampled_video_names + [sampled_video_names[-1]] * num_repeat_image
            sampled_image_names = sampled_image_names + [sampled_image_names[-1]] * num_repeat_image
            video_idxs = video_idxs + [video_idxs[-1]] * num_repeat_image
            local_time_idxs = local_time_idxs + [local_time_idxs[-1] + (rep_t+1) for rep_t in range(num_repeat_image)]
            global_time_idxs = global_time_idxs + [global_time_idxs[-1] + (rep_t+1) for rep_t in range(num_repeat_image)
                                                   ]
        path_info = np.stack([sampled_video_names, sampled_image_names], axis=1)
        index_info = np.stack([video_idxs, local_time_idxs, global_time_idxs], axis=1)
        
        return path_info, index_info

    def _sample_synced_video(self, total_num_image, num_image, num_unsynced_video, \
        num_synced_video, rng, scene_info):
        """
            sample synced dynamic video
            Returns:
                path_info: array, [N,2], last dim [video_name, image_name]
                index_info: array, [N, 3], last dim [video_idx, local_time_idx, global_time_idx]
        """
        # note: each video has same time start, we only sample one clip (for all video)
        scene_num_video = scene_info.get("num_video", None)
        scene_video_names = scene_info.get("video_names", None)
        if scene_num_video is None:
            scene_num_video = self.dataset_info.get("num_video", None)
        assert scene_num_video is not None, "data_sampling for synced dynamic video must provide num_video in scene_info/dataset_info"
        
        assert scene_video_names is not None and len(scene_video_names)==scene_num_video, \
            "data_sampling for synced dynamic video must provide video_names in scene_info, \
                and the length must be equal to num_video"

        num_sampled_video = num_synced_video if scene_num_video >= num_synced_video else scene_num_video
        sampled_video_names_ = rng.choice(scene_video_names, num_sampled_video, replace=False)
        num_video_frame = scene_info.get("num_video_frame", None)
        if num_video_frame is None:
            num_video_frame = self.dataset_info.get("num_video_frame", None)
        assert num_video_frame is not None, "data_sampling for synced dynamic video must provide num_video_frame in scene_info/dataset_info"
        num_frame_interval_range = self.data_sampling_info[DataSamplingType.SyncedDynamicVideo].get("num_frame_interval_range", (1,2))
        num_frame_interval = rng.integers(num_frame_interval_range[0], num_frame_interval_range[1]+1)   

        num_clip_frame_ = total_num_image // num_synced_video # note: clip_frame MUST be divisible by num_synced_video
        frame_idxs_list = self._sample_frame_idx_from_video(num_video_frame, [num_clip_frame_], num_frame_interval, 1, rng)
        sampled_image_names_ = [f"{frame_idx}" for frame_idx in frame_idxs_list[0]]

        # if not enough, repeat the last image of all video clip
        num_repeat_video = num_synced_video - num_sampled_video
        num_repeat_images = num_clip_frame_ - len(sampled_image_names_)
        
        sampled_video_names = np.concatenate([sampled_video_names_, np.repeat(sampled_video_names_[-1], num_repeat_video)])
        sampled_image_names = np.concatenate([sampled_image_names_, np.repeat(sampled_image_names_[-1], num_repeat_images)])

        sampled_video_names = sampled_video_names.reshape(-1, 1).repeat(num_clip_frame_, axis=1)  
        sampled_video_names = sampled_video_names.flatten()
        sampled_image_names = sampled_image_names.reshape(1, -1).repeat(num_synced_video, axis=0)
        sampled_image_names = sampled_image_names.flatten()
        video_idx = np.arange(num_synced_video).reshape(-1, 1).repeat(num_clip_frame_, axis=1)
        video_idx = video_idx.flatten()
        time_idx = np.arange(num_clip_frame_).reshape(1, -1).repeat(num_synced_video, axis=0)
        time_idx = time_idx.flatten()

        path_info = np.stack([sampled_video_names, sampled_image_names], axis=1)
        index_info = np.stack([video_idx, time_idx, time_idx], axis=1) # synced videos share time_idx
        return path_info, index_info

    def _sample_hybrid_video(self, total_num_image, num_image, num_unsynced_video, \
        num_synced_video, rng, scene_info):
        """
            sample hybrid dynamic video, scene_info: oribit_video_info transalte to num_image and image_names first
            Returns:
                path_info: array, [N,2], last dim [video_name, image_name]
                index_info: array, [N, 3], last dim [video_idx, local_time_idx, global_time_idx],
        """
        
        num_static_image = num_image
        num_dynamic_image = total_num_image - num_static_image

        # note
        path_info_oribit, index_info_oribit = self._sample_static_image(num_static_image, num_static_image, num_unsynced_video, \
            num_synced_video, rng, scene_info)

        if num_unsynced_video > num_synced_video:
            path_info_video, index_info_video = self._sample_unsynced_video(num_dynamic_image, 0, num_unsynced_video, \
                num_synced_video, rng, scene_info)        
        else:
            path_info_video, index_info_video = self._sample_synced_video(num_dynamic_image, 0, num_unsynced_video, \
                num_synced_video, rng, scene_info)
        
        # change video_idx, keep local_time_idx unchanged, change global_time_idx
        index_info_video[..., 0] += num_static_image
        index_info_video[..., 2] += num_static_image

        path_info = np.concatenate([path_info_oribit, path_info_video], axis=0)
        index_info = np.concatenate([index_info_oribit, index_info_video], axis=0)  
        return path_info, index_info
        
    def _efficient_sample_image(self, data_sampling_type, total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info):
        """
            sample image for each data_sampling_type according to scene_info
            repeat if necessary
        """
        if data_sampling_type == DataSamplingType.StaticImage:
            path_info, index_info = self._sample_static_image(total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info)
        elif data_sampling_type == DataSamplingType.UnsyncedDynamicVideo:
            path_info, index_info = self._sample_unsynced_video(total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info)
        elif data_sampling_type == DataSamplingType.SyncedDynamicVideo:
            path_info, index_info = self._sample_synced_video(total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info)
        elif data_sampling_type == DataSamplingType.HybridDynamicVideo:
            path_info, index_info = self._sample_hybrid_video(total_num_image, num_image, num_unsynced_video, \
            num_synced_video, rng, scene_info)
        else:
            raise ValueError(f"not implemented data_sampling_type: {data_sampling_type}")
        
        # Uniformly add image_idx as the first column for all sampling methods
        # index_info shape: [N, 3] -> [N, 4], where the first column is image_idx
        image_idx = np.arange(len(index_info))[:, None]
        index_info = np.concatenate([image_idx, index_info], axis=-1)
        
        return path_info, index_info

    
    @staticmethod
    def _sample_frame_idx_from_video(num_video_frame, num_clip_frame_list, num_frame_interval, num_clip_interval, rng):
        """
            Sample frame_idx from video
            Args:
                num_video_frame: int, number of frames in video 
                num_clip_frame_list: List[int], number of frames in each clip
                num_frame_interval: int, interval between frames
                num_clip_interval: int, interval between clips
                rng: numpy.random.default_rng, random number generator
            Returns:
                frame_idxs_list: List[List[int]], frame indices for each clip separately
        """

        # Step 1: Construct fixed sequence starting from 0
        frame_idxs_list = []
        current_pos = 0
        
        for clip_idx, num_clip_frame in enumerate(num_clip_frame_list):
            # Generate frame indices for current clip
            clip_frame_idxs = []
            for frame_idx in range(num_clip_frame):
                frame_pos = current_pos + frame_idx * num_frame_interval
                # Break if frame index exceeds video length
                if frame_pos >= num_video_frame:
                    break
                clip_frame_idxs.append(frame_pos)
            
            # If no valid frames in this clip, break
            if len(clip_frame_idxs) == 0:
                break
                
            frame_idxs_list.append(clip_frame_idxs)
            
            # Move to next clip position (add clip interval)
            current_pos = clip_frame_idxs[-1] + num_clip_interval
            
            # Break if next clip would start beyond video length
            if current_pos >= num_video_frame:
                break
        
        # Step 2: Calculate maximum possible offset
        if len(frame_idxs_list) == 0:
            return []
            
        # Find the maximum frame index across all clips
        all_frame_idxs = [idx for clip_idxs in frame_idxs_list for idx in clip_idxs]
        max_frame_idx = max(all_frame_idxs)
        max_offset = num_video_frame - 1 - max_frame_idx
        
        # Step 3: Randomly sample offset and add to all indices
        if max_offset > 0:
            offset = rng.integers(0, max_offset + 1)
            frame_idxs_list = [[idx + offset for idx in clip_idxs] for clip_idxs in frame_idxs_list]
        
        return frame_idxs_list

    ## trajectory
    def _merge_trajectories(self, trajectory_arrays, merge_strategy='overlay'):
        """Merge multiple trajectory arrays using vectorized operations (no loops).
        
        Args:
            *trajectory_arrays: Multiple arrays of shape [im, t, h, w, 3]
            merge_strategy: 'overlay' (later arrays override earlier ones) or 
                        'fill' (only fill NaN positions)
            dynamic_weight: float, weight of foreground trajectories
            
        Returns:
            merged_trajectories: Array of shape [im, t, h, w, 3] - merged result
            trajectory_foreground: Array of shape [im, t, h, w] - foreground mask for each trajectory
        """
        if len(trajectory_arrays) == 0:
            raise ValueError("At least one trajectory array is required")
        
        merged = trajectory_arrays[0].copy()
        trajectory_foreground = np.zeros_like(merged[..., 0]).astype(bool)
        for trajectory in trajectory_arrays[1:]:
            if merge_strategy == 'overlay':
                # Later arrays override earlier ones where they have valid values
                # Check if all 3 coordinates are valid (not NaN)
                valid_mask = ~np.isnan(trajectory).all(axis=-1)  # [im, t, h, w]
                merged[valid_mask] = trajectory[valid_mask]
                trajectory_foreground[valid_mask] = True
            elif merge_strategy == 'fill':
                # Only fill positions where merged array has NaN values
                merged_nan_mask = np.isnan(merged).all(axis=-1)  # [im, t, h, w] 
                trajectory_valid_mask = ~np.isnan(trajectory).all(axis=-1)  # [im, t, h, w]
                fill_mask = merged_nan_mask & trajectory_valid_mask  # [im, t, h, w]
                merged[fill_mask] = trajectory[fill_mask]
                trajectory_foreground[fill_mask] = True
                
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        return merged, trajectory_foreground
    
    def _fill_trajectories_from_pts3d(self, pts3d_per_frame, camera_intrinsics, camera_poses, 
                                    image_shape, frame_to_image_mapping, static_masks=None,
                                    return_depthmaps=False):
        """Fill trajectories from per-frame 3D points using vectorized operations.
        
        This method efficiently projects all 3D points to image coordinates using batch
        operations and fills trajectory arrays. For static points, the same 3D position 
        is replicated across all time steps.
        
        Args:
            pts3d_per_frame: Array of shape [t, n, 3] - 3D points for each frame
            camera_intrinsics: Array of shape [im, 3, 3] - camera intrinsic matrices
            camera_poses: Array of shape [im, 4, 4] - camera poses (c2w)
            image_shape: Tuple (h, w) - height and width of images
            frame_to_image_mapping: Array of shape [im] - maps each image to its global frame idx
            static_masks: Optional array of shape [im, h, w] - binary masks for static regions
            return_depthmaps: Bool - whether to also return depth maps
            
        Returns:
            trajectories: Array of shape [im, t, h, w, 3] - filled trajectory array
            depthmaps: Optional array of shape [im, h, w] - depth maps (if return_depthmaps=True)
        """
        num_images = len(camera_intrinsics)
        num_frames, max_pts = pts3d_per_frame.shape[:2]
        
        # Select points for each image based on frame mapping
        im_pts3d = pts3d_per_frame[frame_to_image_mapping]  # [im, n, 3]
        
        # Batch project all points to image coordinates
        uv_coords, valid_mask, depths = world_to_pixel_coordinates(
            im_pts3d, camera_intrinsics, camera_poses, image_shape,
            has_batch=True, return_depth=True
        )  # [im, n, 2], [im, n], [im, n]

        uv_coords[~valid_mask] = -1
        uv_coords = np.round(uv_coords).astype(int)
        
        # Get valid projected data
        valid_uv = uv_coords[valid_mask]  # [num_valid, 2]
        valid_depths = depths[valid_mask]  # [num_valid]
        valid_pts3d = im_pts3d[valid_mask]  # [num_valid, 3]
        
        # Create image and time index arrays for valid points
        im_index_expanded = np.arange(num_images)[:, None].repeat(max_pts, axis=1)  # [im, n]
        time_index_expanded = frame_to_image_mapping[:, None].repeat(max_pts, axis=1)  # [im, n]
        
        valid_im_indices = im_index_expanded[valid_mask]  # [num_valid]
        valid_time_indices = time_index_expanded[valid_mask]  # [num_valid]
        
        # Initialize trajectory array
        trajectories = np.full(
            (num_images, num_frames, *image_shape, 3), np.nan, dtype=np.float32
        )
        
        # Fill trajectories at current time step for valid points
        trajectories[valid_im_indices, valid_time_indices, valid_uv[:, 1], valid_uv[:, 0]] = valid_pts3d
        
        # Handle static mask if provided
        if static_masks is not None:
            static_masks_bool = static_masks.astype(bool)  # [im, h, w]
            # Check which valid points are in static regions
            is_valid_pts3d_static = static_masks_bool[valid_im_indices, valid_uv[:, 1], valid_uv[:, 0]]
            
            if is_valid_pts3d_static.any():
                static_im_indices = valid_im_indices[is_valid_pts3d_static]
                static_uv = valid_uv[is_valid_pts3d_static]
                static_pts3d = valid_pts3d[is_valid_pts3d_static]
                
                # Fill static points across all time steps
                trajectories[static_im_indices, :, static_uv[:, 1], static_uv[:, 0]] = static_pts3d[:, None, :]
        
        if return_depthmaps:
            # Fill depth maps
            depthmaps = np.full((num_images, *image_shape), np.nan, dtype=np.float32)
            depthmaps[valid_im_indices, valid_uv[:, 1], valid_uv[:, 0]] = valid_depths
            return trajectories, depthmaps
        
        return trajectories
    
    def _fill_trajectories_from_trajectory(self, full_trajectories, camera_intrinsics, camera_poses, 
                                        image_shape, frame_to_image_mapping):
        """Fill trajectories from full trajectory data using vectorized operations (no loops).
        
        This method handles trajectory data in format [t, t, k, 3] where each reference frame
        has its own complete trajectory data with different visibility constraints.
        
        Args:
            full_trajectories: Array of shape [t, t, k, 3] - complete trajectory data
            camera_intrinsics: Array of shape [im, 3, 3] - camera intrinsic matrices
            camera_poses: Array of shape [im, 4, 4] - camera poses (c2w)
            image_shape: Tuple (h, w) - height and width of images
            frame_to_image_mapping: Array of shape [im] - maps each image to its global frame idx
            
        Returns:
            trajectories: Array of shape [im, t, h, w, 3] - filled trajectory array
        """
        num_images = len(camera_intrinsics)
        num_frames = full_trajectories.shape[1]  # Second dimension is temporal
        
        # Select trajectory data for each image based on frame mapping
        im_trajectories = full_trajectories[frame_to_image_mapping]  # [im, t, k, 3]
        
        # Extract current frame points for projection using advanced indexing
        im_indices = np.arange(num_images)
        frame_indices = frame_to_image_mapping
        im_current_pts3d = im_trajectories[im_indices, frame_indices]  # [im, k, 3]
        
        # Batch project current frame points to image coordinates
        uv_coords, valid_mask = world_to_pixel_coordinates(
            im_current_pts3d, camera_intrinsics, camera_poses, image_shape,
            has_batch=True, return_depth=False
        )  # [im, k, 2], [im, k]
        uv_coords[~valid_mask] = -1
        uv_coords = np.round(uv_coords).astype(int)
        
        # Get valid projected data
        valid_uv = uv_coords[valid_mask]  # [num_valid, 2]
        
        # Get valid trajectory data using np.where
        valid_im_indices, valid_k_indices = np.where(valid_mask)  # [num_valid], [num_valid]
        valid_trajectories = im_trajectories[valid_im_indices, :, valid_k_indices]  # [num_valid, t, 3]
        
        # Initialize trajectory array
        trajectories = np.full(
            (num_images, num_frames, *image_shape, 3), np.nan, dtype=np.float32
        )
        
        # Fill all time steps for valid vehicle trajectory points
        if len(valid_trajectories) > 0:
            trajectories[valid_im_indices, :, valid_uv[:, 1], valid_uv[:, 0]] = valid_trajectories
        
        return trajectories

    ## main func
    def _get_views(self, idx, data_sampling_type, total_num_image, num_image, num_unsynced_video, \
            num_synced_video, aspect_ratio, resolution, rng):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        # TODO: we apply several transform here, e.g. data_normalization, data augmentation, image_augmentation, etc.
        # we will handle both image, depth_map, mask, trajectory(2d and 3d)
        # for simplicity, we compute full trajectory for each sample, default None    
        # NOTE:
        #   | depth_map | trajectory | Action                                              |
        #   |-----------|------------|-----------------------------------------------------|
        #   | None      | Available  | Project trajectory to obtain depth (in get_views).  |
        #   | Available | None       | Project depth_map to obtain one-timestep trajectory |
        #   | Available | Available  | If merge_traj is True, merge both (priority)        |
        #   | None      | None       | Pass                                                |

        # idx here is local dataset idx
        assert isinstance(idx, (tuple, list, np.ndarray)), "only support custom sampler now"
        idx, data_sampling_type, total_num_image, num_image, num_unsynced_video, \
            num_synced_video, aspect_ratio = idx

        # assert self.seed is None, "we need randomness for scene during training"
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        resolution = self._adjust_resolution(self._resolution, aspect_ratio, self.patch_size)

        # all images should be loaded here (repeating)
        views = self._get_views(idx, data_sampling_type, total_num_image, num_image, num_unsynced_video, \
            num_synced_video, resolution, self._rng)
        
        
        if self._seq_aug_crop:
            if self._aug_crop > 1:
                self.delta_target_resolution = self._rng.integers(0, self._aug_crop)
            elif 0 < self._aug_crop < 1:
                delta_target_ratio = self._rng.random() * (1. / self._aug_crop - 1.)
                self.delta_target_resolution = (np.array(resolution) * delta_target_ratio).astype("int")

        # camera normalization: normalized by first camera pose, implement for trajectory and camera pose
        if self._normalize_camera_pose:
            views = self._normalize_by_first_camera_pose(views)
        
        # point normalization: normalized by point cloud average scale, implement for depth, trajectory and camera_pose
        # note: 1) should only take valid points, 2)TODO:what if the video has only rgb pose, no pointcloud/depth map?

        # process_trajectory
        # merge depth_map_pts and trajectory here
        if "depth" in views or "trajectory" in views:
            views = self._process_trajectory(views)
        
        if self._normalize_point_cloud:
            views = self._normalized_by_pointcloud_scale(views)

        # process_valid_mask
        if "valid_mask" not in views:
            valid_mask = np.ones_like(views["image"][..., 0]).astype(bool)
            views["valid_mask"] = valid_mask
        
        # process_image, to_tensor
        views["image"] = to_tensor(views["image"])
        # image augmentation
        if self._aug_image:
            views = self._aug_all_view_image(views)  
        
        # misc compute time_idx2image_idx and add rng to scene_meta
        views["time_idx2image_idx"] = self._compute_time_idx2image_idx(views["image_info"])
        views["scene_meta"]["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        views["scene_meta"]["path_info"] = views["scene_meta"]["path_info"].tolist()
        
        return views
    
    ## misc
    def _set_resolution(self, resolution):
        assert resolution is not None, "undefined resolution"
        if isinstance(resolution, int):
            width = height = resolution
        else:
            width, height = resolution
        assert isinstance(
            width, int
        ), f"Bad type for {width=} {type(width)=}, should be int"
        assert isinstance(
            height, int
        ), f"Bad type for {height=} {type(height)=}, should be int"
        self._resolution = (width, height)

    @staticmethod
    def _adjust_resolution(resolution, aspect_ratio, patch_size, only_h=True):
        W, H = resolution
        adjusted_H = H * aspect_ratio[1]

        remainder_h = adjusted_H % patch_size
        if remainder_h != 0:
            adjusted_H += patch_size - remainder_h
        if not only_h:
            adjusted_W = W * aspect_ratio[0]
            remainder_w = adjusted_W % patch_size
            if remainder_w != 0:
                adjusted_W += patch_size - remainder_w
        else:
            adjusted_W = W
        
        return (int(adjusted_W), int(adjusted_H))
    
    @staticmethod
    def _pad_and_stack_arrays(array_list, max_size=None, pad_value=np.nan, axis=-1):
        """
        Pad arrays along the specified axis to max_size, then stack along a new first dimension.

        Parameters:
        - array_list: list of np.ndarray, all with the same number of dimensions
        - max_size: int or None, max length along axis to pad to; if None, use max length in array_list
        - pad_value: value to use for padding
        - axis: int, axis along which to pad arrays

        Returns:
        - stacked np.ndarray of shape (batch_size, ..., max_size, ...)
        where max_size is the padded size along the specified axis
        """
        if not array_list:
            return np.array([])

        ndim = array_list[0].ndim
        axis = axis if axis >= 0 else ndim + axis

        # Determine max size along axis if not given
        if max_size is None:
            max_size = max(arr.shape[axis] for arr in array_list)

        batch_size = len(array_list)

        # Build output shape: insert max_size at axis+1 (because batch dim is added at front)
        ref_shape = list(array_list[0].shape)
        ref_shape[axis] = max_size
        output_shape = [batch_size] + ref_shape

        # Create output array filled with pad_value
        result = np.full(output_shape, pad_value, dtype=array_list[0].dtype)

        for i, arr in enumerate(array_list):
            length = arr.shape[axis]
            length = min(length, max_size)

            # Build slices for result: batch dim = i, axis dimension = 0:length, others = slice(None)
            result_slices = [i] + [slice(None)] * ndim
            result_slices[axis + 1] = slice(0, length)

            # Build slices for arr: axis dimension = 0:length, others = slice(None)
            arr_slices = [slice(None)] * ndim
            arr_slices[axis] = slice(0, length)

            # Copy data into result
            result[tuple(result_slices)] = arr[tuple(arr_slices)]

        return result

    
    @staticmethod
    def _compute_time_idx2image_idx(image_info):
        """Compute time_idx2image_idx mapping from image_info using vectorized operations.
        
        Args:
            image_info: Array of shape [im, 4] containing:
                    (image_idx, video_idx, local_time_idx, global_time_idx)
        
        Returns:
            time_idx2image_idx: Dict mapping global_time_idx to list of image indices
        """
        global_time_indices = image_info[:, 3].astype(int)  # Extract global_time_idx column
        image_indices = np.arange(len(image_info))  # [0, 1, 2, ..., im-1]
        
        time_idx2image_idx = {}
        unique_times = np.unique(global_time_indices)
        
        for time_idx in unique_times:
            mask = global_time_indices == time_idx
            time_idx2image_idx[int(time_idx)] = image_indices[mask]
        
        return time_idx2image_idx

        
    
    @staticmethod
    def _check_sample(views):
        """Check the value of views, some tensor should not be nan while depth/trajectory should not be all nan"""
        pass



class BaseTestDataset(BaseDataset):
    """Base test dataset
        scenes: list of scene_info
            scene_info:
                data_sampling_type: str
                others for data_sampling
    """

    def _get_static_image(self, video_names, image_names, is_synced=None):
        
        num_image = len(image_names[0])
        video_names = video_names * num_image
        path_info = np.stack([video_names, image_names[0]], axis=-1)
        # tmp_debug, modify
        index_info = np.stack([np.arange(num_image).astype(np.int64), np.zeros(num_image).astype(np.int64), np.arange(num_image).astype(np.int64)], axis=1)

        return path_info, index_info
    
    def _get_unsynced_video(self, video_names, image_names, is_synced=None):

        num_image = sum(len(image_names[i]) for i in range(len(image_names)))
        path_info = []
        local_time_idxs = []
        video_idxs = []
        for video_idx, (video_name, image_names_per_video) in enumerate(zip(video_names, image_names)):
            path_info.append(np.stack([np.array([video_name] * len(image_names_per_video)), np.array(image_names_per_video)], axis=-1))
            video_idxs.append(np.array([video_idx] * len(image_names_per_video)))
            local_time_idxs.append(np.arange(len(image_names_per_video)))
        global_time_idxs = np.arange(num_image)
        path_info = np.concatenate(path_info, axis=0)
        video_idxs = np.concatenate(video_idxs, axis=0)
        local_time_idxs = np.concatenate(local_time_idxs, axis=0)
        index_info = np.stack([video_idxs, local_time_idxs, global_time_idxs], axis=1)

        return path_info, index_info
    
    def _get_synced_video(self, video_names, image_names, is_synced=None):

        assert len(set([len(image_names[i]) for i in range(len(image_names))])) == 1, \
            "All videos should have the same number of images"
        num_time = len(image_names[0])
        path_info = []
        video_idxs = []
        for video_idx, (video_name, image_names_per_video) in enumerate(zip(video_names, image_names)):
            path_info.append(np.stack([np.array([video_name] * len(image_names_per_video)), np.array(image_names_per_video)], axis=-1))
            video_idxs.append(np.array([video_idx] * len(image_names_per_video)))
        path_info = np.concatenate(path_info, axis=0)
        video_idxs = np.concatenate(video_idxs, axis=0)
        index_info = np.stack([video_idxs, np.arange(num_time), np.arange(num_time)], axis=1)

        return path_info, index_info
    
    def _get_hybrid_video(self, video_names, image_names, is_synced=None):

        path_info_oribit, index_info_oribit = self._get_static_image(video_names[0:1], image_names[0:1])
        video_names = video_names[1:]
        image_names = image_names[1:] 
        if is_synced:
            path_info_video, index_info_video = self._get_synced_video(video_names, image_names, is_synced)
        else:
            path_info_video, index_info_video = self._get_unsynced_video(video_names, image_names, is_synced)        
                
        # change video_idx, keep local_time_idx unchanged, change global_time_idx
        index_info_video[..., 0] += len(path_info_oribit)
        index_info_video[..., 2] += len(path_info_oribit)

        path_info = np.concatenate([path_info_oribit, path_info_video], axis=0)
        index_info = np.concatenate([index_info_oribit, index_info_video], axis=0)

        return path_info, index_info

    def _get_image_info(self, data_sampling_type, video_names, image_names, is_synced=None):
        """
            get_sample_info for the scene
            Args:

                data_sampling_type: str
                is_synced: for hybrid setting
                video_names: list[str]
                image_names: list[list[str]], first list is video, second list is image_name
                    if static_image, use image_paths[0]
                    if unsynced/synced video, loop all videos
                    if hybrid video, image_paths[0] is oribit video, rest image is synced/unsynced video
                    
            Return:
                path_info: np.ndarray of shape [im, 2], (video_name, image_name)
                index_info: np.ndarray of shape [im, 4], (image_idx, video_idx, local_time_idx, global_time_idx)
        """

        if data_sampling_type == DataSamplingType.StaticImage:
            path_info, index_info = self._get_static_image(video_names, image_names, is_synced)
        elif data_sampling_type == DataSamplingType.UnsyncedDynamicVideo:
            path_info, index_info = self._get_unsynced_video(video_names, image_names, is_synced)
        elif data_sampling_type == DataSamplingType.SyncedDynamicVideo:
            path_info, index_info = self._get_synced_video(video_names, image_names, is_synced)
        elif data_sampling_type == DataSamplingType.HybridDynamicVideo:
            path_info, index_info = self._get_hybrid_video(video_names, image_names, is_synced)
    
        else:
            raise ValueError(f"not implemented data_sampling_type: {data_sampling_type}")
        
        # Uniformly add image_idx as the first column for all sampling methods
        # index_info shape: [N, 3] -> [N, 4], where the first column is image_idx
        image_idx = np.arange(len(index_info))[:, None]
        index_info = np.concatenate([image_idx, index_info], axis=-1)
        
        return path_info, index_info


    def __getitem__(self, idx):
    
        # note: do not use random when eval
        # assert self.seed is not None, "seed should be set for validation"

        # set-up the rng
        # if self.seed:  # reseed for each __getitem__
        #     self._rng = np.random.default_rng(seed=self.seed + idx)
        # elif not hasattr(self, "_rng"):
        #     seed = torch.randint(0, 2**32, (1,)).item()
        #     self._rng = np.random.default_rng(seed=seed)
        # TODO: comment this, this is for debug_fit
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx = idx[0]

        # all images should be loaded here (repeating)
        views = self._get_views(idx)
           
        # note: we process validation data the same as training data for evaluation metric

        # camera normalization: normalized by first camera pose, implement for trajectory and camera pose
        if self._normalize_camera_pose:
            views = self._normalize_by_first_camera_pose(views)
        
        # point normalization: normalized by point cloud average scale, implement for depth, trajectory and camera_pose
        # note: 1) should only take valid points, 2)TODO:what if the video has only rgb pose, no pointcloud/depth map?

        # process_trajectory
        # merge depth_map_pts and trajectory here
        if "depth" in views or "trajectory" in views:
            views = self._process_trajectory(views)
        
        if self._normalize_point_cloud:
            views = self._normalized_by_pointcloud_scale(views)

        # process_valid_mask
        if "valid_mask" not in views:
            valid_mask = np.ones_like(views["image"][..., 0]).astype(bool)
            views["valid_mask"] = valid_mask
        
        # process_image, to_tensor
        views["image"] = to_tensor(views["image"])
        # note: we do not appply image aug here
        # image augmentation
        # if self._aug_image:
        #     views = self._aug_all_view_image(views)  
        
        # misc compute time_idx2image_idx and add rng to scene_meta
        views["time_idx2image_idx"] = self._compute_time_idx2image_idx(views["image_info"])
        views["scene_meta"]["path_info"] = views["scene_meta"]["path_info"].tolist()

        return views