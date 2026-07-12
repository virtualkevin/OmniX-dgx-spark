import torch
import numpy as np


from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection

## point trajectory
from einops import rearrange
from src.utils.projection import closed_form_inverse_se3, world_to_pixel_coordinates
from src.visualization.drawing.point import draw_points

def render_point_cloud(
    points,
    intrinsics,
    camera_poses,
    colors=None,
    image_shape=None,
    return_depth=True,
):
    """
    Pytorch3D render point cloud with input parameters in opencv style
    Params:
    points: Tensor [..., 3]
    colors: Tensor [..., 3]
    intrinsics: Tensor [N, 3, 3]
    camera_poses: Tensor [N, 4, 4]
    image_shape: tuple [H, W]

    Return: 
        Image: Tensor [N, H, W, 3]
        Depth: Tensor [N, H, W, 1]
    """
  
    with torch.cuda.amp.autocast(enabled=False):
        points = points.reshape(-1, 3)
        if colors is not None:
            colors = colors.reshape(-1, 3)
        T_world_to_cameras = closed_form_inverse_se3(camera_poses)
        R_wc, t_wc = T_world_to_cameras[:, :3, :3], T_world_to_cameras[:, :3, 3]

        device = points.device
        opencv_converted_cameras = cameras_from_opencv_projection(
            R_wc,
            t_wc,
            intrinsics,
            torch.tensor(image_shape, device=device).unsqueeze(0).repeat(camera_poses.shape[0], 1)
        )

        # pytorch3d 的点云格式
        batch_points = [points for _ in range(len(camera_poses))]
        batch_colors = None
        if colors is not None:
            batch_colors = [colors for _ in range(len(camera_poses))]
        point_cloud = Pointclouds(points=batch_points, features=batch_colors)

        raster_settings = PointsRasterizationSettings(
            image_size=tuple(image_shape), 
            radius=0.01,    # 你可调整这个效果
            points_per_pixel=16
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=opencv_converted_cameras,
                raster_settings=raster_settings
            ),
            compositor=AlphaCompositor(background_color=(1, 1, 1))  # 白底
        )

        rgb_images = renderer(point_cloud) # (N, H, W, 3)
        if not return_depth:
            return rgb_images
        else:
            fragments = renderer.rasterizer(point_cloud)
            depth_maps = fragments.zbuf[..., 0] # (N, H, W)

            return rgb_images, depth_maps



def render_all_pointcloud_trajectory(
    trajectory,
    intrinsics,
    camera_poses,
    images=None,
    return_depth=False,
    max_points=100000, # 全局最大点数
):
    """
    Args:
        trajectory: Tensor [im, t, h, w, 3]
        intrinsics: Tensor [im, 3, 3]
        camera_poses: Tensor [im, 4, 4]
        images: Tensor [im, h, w, 3]
        return_depth: bool
    """ 
    im, t, h, w, _ = trajectory.shape
    device = trajectory.device
    image_shape = (h, w)
    
    # -----------------------------------------------------------
    # 1. 准备全局索引 (一次性准备好，保证视频每一帧采样同样位置的点)
    # -----------------------------------------------------------
    total_possible_points = im * h * w
    
    # 如果总点数太多，就生成一个全局的随机索引
    if total_possible_points > max_points:
        # 从 [0, im*h*w] 中随机选 max_points 个
        sample_indices = torch.randperm(total_possible_points, device=device)[:max_points]
    else:
        sample_indices = None

    # -----------------------------------------------------------
    # 2. 准备颜色 (静态的，只做一次)
    # -----------------------------------------------------------
    if images is not None:
        # [im, h, w, 3] -> [im*h*w, 3]
        colors_flat = images.reshape(-1, 3)
    else:
        colors_flat = torch.ones((total_possible_points, 3), device=device)

    # 全局采样: 得到 [max_points, 3]
    if sample_indices is not None:
        colors_input = colors_flat[sample_indices]
    else:
        colors_input = colors_flat
    
    # 注意：这里不需要 expand！
    # 因为你的 render_point_cloud 内部会根据 camera_poses 的数量自动复用这一份颜色

    pointcloud_images = []
    pointcloud_depths = [] if return_depth else None

    # -----------------------------------------------------------
    # 3. 时间循环
    # -----------------------------------------------------------
    for time_idx in range(t):
        # A. 融合所有视角的点: [im, h, w, 3] -> [im*h*w, 3]
        traj_flat = trajectory[:, time_idx].reshape(-1, 3)
        
        # B. 采样: [max_points, 3]
        if sample_indices is not None:
            points_input = traj_flat[sample_indices]
        else:
            points_input = traj_flat

        # C. 清洗: 必须做！防止 NaN 搞崩 CUDA
        # 你的 render_point_cloud 里没有处理 NaN，所以必须在这里处理
        if torch.isnan(points_input).any() or torch.isinf(points_input).any():
             points_input = torch.nan_to_num(points_input, nan=0.0, posinf=1e5, neginf=-1e5)

        # D. 调用渲染器
        # 关键点：
        # points_input 是 [max_points, 3]
        # camera_poses 是 [im, 4, 4]
        # 你的 render_point_cloud 会自动让这 im 个相机都渲染这份 points_input
        if not return_depth:
            pc_image = render_point_cloud(
                points_input,        # [max_points, 3]
                intrinsics=intrinsics,
                camera_poses=camera_poses,
                colors=colors_input, # [max_points, 3]
                image_shape=image_shape,
                return_depth=return_depth,
            )
        else:
            pc_image, pc_depth = render_point_cloud(
                points_input,
                intrinsics=intrinsics,
                camera_poses=camera_poses,
                colors=colors_input,
                image_shape=image_shape,
                return_depth=return_depth,
            )
            pointcloud_depths.append(pc_depth)
            
        pointcloud_images.append(pc_image)

    
    pointcloud_images = torch.stack(pointcloud_images, dim=1) # [im, t, h, w, c]
    
    if return_depth:
        pointcloud_depths = torch.stack(pointcloud_depths, dim=1).squeeze(-1)
        return pointcloud_images, pointcloud_depths
    
    return pointcloud_images
    
def render_per_image_pointcloud_trajectory(
    trajectory,
    intrinsics,
    camera_poses,
    images=None,
    return_depth=False
):
    """Render point cloud trajectory, use pred point cloud from single images
        Args:
            trajectory: Tensor [im t h w xyz]
            intrinsics: Tensor [im 3 3]
            camera_poses: Tensor [im 4 4]
            images: Tensor [im h w 3]
            return_depth: bool
        Returns:
            pointcloud_images: Tensor [im t h w 3]
            pointcloud_depths:[optional] Tensor [im t h w]
    """ 
    im, t, h, w, _ = trajectory.shape
    image_shape = (h, w)

    pointcloud_images = []
    if return_depth:
         pointcloud_depths = []
    
    for image_idx in range(im):
        for time_idx in range(t):
            trajectory_single_image_single_time = trajectory[image_idx, time_idx]
            if not return_depth:
                pc_image= render_point_cloud(
                    trajectory_single_image_single_time,
                    intrinsics=intrinsics[image_idx:image_idx+1],
                    camera_poses=camera_poses[image_idx:image_idx+1],
                    colors=images[image_idx:image_idx+1],
                    image_shape=image_shape,
                    return_depth=return_depth,
                )
            else:
                pc_image, pc_depth= render_point_cloud(
                    trajectory_single_image_single_time,
                    intrinsics=intrinsics[image_idx:image_idx+1],
                    camera_poses=camera_poses[image_idx:image_idx+1],
                    colors=images[image_idx],
                    image_shape=image_shape,
                    return_depth=return_depth,
                )
            pointcloud_images.append(pc_image)
            if return_depth:
                pointcloud_depths.append(pc_depth)

    pointcloud_images = torch.cat(pointcloud_images, dim=0) # [(im t) h w c]
    pointcloud_images = rearrange(pointcloud_images, '(im t) h w c -> im t h w c', im=im)
    if not return_depth:
        return pointcloud_images
    else:
        pointcloud_depths = torch.cat(pointcloud_depths, dim=0).squeeze(-1) # [(im t) h w]
        pointcloud_depths = rearrange(pointcloud_depths, '(im t) h w -> im t h w', im=im)
        return pointcloud_images, pointcloud_depths

def draw_per_image_pointcloud_trajectory(
    trajectory,
    intrinsics,
    camera_poses,
    render_images=None,
    sampled_ratio=0.1,
):
    """Draw per-image pointcloud trajectory on per-image renderings
        Args:
            trajectory: Tensor [im t h w xyz]
            intrinsics: Tensor [im 3 3]
            camera_poses: Tensor [im 4 4]
            render_images: Tensor [im t h w 3]
            sampled_ratio: float
    """

    im, t, h, w, _ = trajectory.shape

    step = int(h * sampled_ratio)
    pointcloud_images_with_traj = []
    for image_idx in range(im):
        color_traj = None
        for time_idx in range(t):
            trajectory_single_image_single_time= trajectory[image_idx, time_idx]
            sampled_trajectory = trajectory_single_image_single_time[step//2::step, step//2::step] # [hh ww xyz]
            if color_traj is None:
                color_generator = torch.Generator(device=sampled_trajectory.device)
                color_generator.manual_seed(42)
                color_traj = torch.rand(
                    sampled_trajectory.shape,           # 同样的形状
                    dtype=sampled_trajectory.dtype,     # 同样的数据类型
                    device=sampled_trajectory.device,   # 同样的设备
                    generator=color_generator           # 使用指定的生成器
                )

            sampled_trajectory_2d, sampled_trajectory_2d_valid_mask = world_to_pixel_coordinates(
                sampled_trajectory,
                camera_intrinsic=intrinsics[image_idx],
                camera_pose=camera_poses[image_idx],
                image_shape=trajectory_single_image_single_time.shape[:2],
                has_batch=False,
                return_depth=False
            ) # [hh ww 2] [hh ww]
            valid_sampled_trajectory_2d = sampled_trajectory_2d[sampled_trajectory_2d_valid_mask]
            valid_color = color_traj[sampled_trajectory_2d_valid_mask]
            image_with_traj = draw_points(
                render_images[image_idx, time_idx],
                valid_sampled_trajectory_2d, 
                valid_color,
                radius=3,
            )
            
            pointcloud_images_with_traj.append(image_with_traj)
    pointcloud_images_with_traj = torch.stack(pointcloud_images_with_traj, dim=0) # [(im t) h w c]
    pointcloud_images_with_traj = rearrange(pointcloud_images_with_traj, '(im t) h w c -> im t h w c', im=im)
    return pointcloud_images_with_traj


    

    
