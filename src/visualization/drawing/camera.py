import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def draw_cameras_with_frustum(camera_poses, colors, res=400, camera_span=1/16, margin_scale=1.8, line_width=2, point_radius=8, font_size=16):
    """
    Draw camera poses with frustums in three orthogonal views using PIL.
    
    Args:
        camera_poses: Camera-to-world transformation matrices [N, 4, 4]
        colors: Camera colors, either (3,) for all cameras or (N, 3) for individual colors
        
    Returns:
        Combined numpy image array (H, W*3, 3) with XY, YZ, XZ views
    """
    N = camera_poses.shape[0]
    
    # Normalize colors to (N, 3)
    if isinstance(colors, (list, tuple)):
        colors = np.array(colors).reshape(1, 3).repeat(N, axis=0)
    elif isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
        if colors.shape == (3,):
            colors = colors.reshape(1, 3).repeat(N, axis=0)
    colors = (colors * 255).astype(int)
    
    # ===== 提取相机中心 =====
    centers = camera_poses[:, :3, 3].cpu().numpy()

    min_vals_scene = centers.min(axis=0)
    max_vals_scene = centers.max(axis=0)
    scene_span = np.linalg.norm(max_vals_scene - min_vals_scene)  # 场景对角线长度

    frustum_scale = scene_span * camera_span
    near = frustum_scale * 0.5
    far = frustum_scale * 1.5

    frustum_corners = compute_frustum_corners(camera_poses, frustum_scale, near, far)

    all_points_3d = np.concatenate([
        centers, 
        frustum_corners['near'].reshape(-1, 3), 
        frustum_corners['far'].reshape(-1, 3)
    ])
    min_vals_3d = all_points_3d.min(axis=0)
    max_vals_3d = all_points_3d.max(axis=0)
    spans_3d = (max_vals_3d - min_vals_3d) * margin_scale
    centers_3d = (max_vals_3d + min_vals_3d) / 2

    # ===== 最大跨度，用于三视图一致性缩放 =====
    global_span = spans_3d.max()
    
    # Draw three views with consistent scaling
    views = []
    view_names = ["XY View", "YZ View", "XZ View"]
    axes_list = [(0, 1), (1, 2), (0, 2)]
    
    for axes, name in zip(axes_list, view_names):
        img = draw_single_view_consistent(centers, frustum_corners, colors, axes, name, 
                                        res, global_span, centers_3d, 
                                        line_width, point_radius, font_size)
        views.append(img)
    
    return np.concatenate(views, axis=1)

def compute_frustum_corners(camera_poses, frustum_scale, near, far):
    """Compute near and far frustum corners for all cameras."""
    device = camera_poses.device
    N = camera_poses.shape[0]
    
    # Define frustum corners in camera coordinate system
    corners_cam = torch.tensor([
        [-frustum_scale, -frustum_scale, 1.0],  # bottom-left
        [ frustum_scale, -frustum_scale, 1.0],  # bottom-right
        [ frustum_scale,  frustum_scale, 1.0],  # top-right
        [-frustum_scale,  frustum_scale, 1.0]   # top-left
    ], device=device, dtype=camera_poses.dtype).repeat(N, 1, 1)
    
    # Transform to world coordinates
    R = camera_poses[:, :3, :3]
    t = camera_poses[:, :3, 3:4]
    
    near_corners = torch.matmul(R, corners_cam.transpose(1, 2) * near).transpose(1, 2) + t.transpose(1, 2)
    far_corners = torch.matmul(R, corners_cam.transpose(1, 2) * far).transpose(1, 2) + t.transpose(1, 2)
    
    return {
        'near': near_corners.cpu().numpy(),
        'far': far_corners.cpu().numpy()
    }

def draw_single_view_consistent(centers, frustum_corners, colors, axes, view_name, res, 
                              global_span, global_center, line_width, point_radius, font_size):
    """Draw cameras in a single 2D projection view with consistent scaling."""
    # Project to 2D using specified axes
    centers_2d = centers[:, axes]
    near_2d = frustum_corners['near'][:, :, axes]
    far_2d = frustum_corners['far'][:, :, axes]
    
    # Use global center and span for this projection
    center_2d = global_center[list(axes)]
    # Create white image
    img = Image.new('RGB', (res, res), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Convert to pixel coordinates using consistent scaling
    def to_pixel(pts):
        return ((pts - (center_2d - global_span/2)) / global_span * (res-1)).astype(int)
    
    centers_pix = to_pixel(centers_2d)
    near_pix = to_pixel(near_2d)
    far_pix = to_pixel(far_2d)
    
    N = len(centers)
    for i in range(N):
        color_rgb = tuple(colors[i])
        
        # Draw frustum lines
        # Near plane rectangle
        for j in range(4):
            draw.line([tuple(near_pix[i, j]), tuple(near_pix[i, (j+1)%4])], 
                     fill=color_rgb, width=line_width)
        
        # Far plane rectangle  
        for j in range(4):
            draw.line([tuple(far_pix[i, j]), tuple(far_pix[i, (j+1)%4])], 
                     fill=color_rgb, width=line_width)
        
        # Lines from center to near corners
        for j in range(4):
            draw.line([tuple(centers_pix[i]), tuple(near_pix[i, j])], 
                     fill=color_rgb, width=line_width)
        
        # Lines from near to far corners
        for j in range(4):
            draw.line([tuple(near_pix[i, j]), tuple(far_pix[i, j])], 
                     fill=color_rgb, width=max(1, line_width//2))
        
        # Draw camera center (circle using ellipse)
        x, y = centers_pix[i]
        draw.ellipse([x-point_radius, y-point_radius, x+point_radius, y+point_radius], 
                    fill=color_rgb)
    
    # Add labels
    draw.text((10, 10), view_name, fill=(0, 0, 0), font=font)
    
    # Camera indices
    for i in range(N):
        x, y = centers_pix[i]
        # Make sure text is within image bounds
        if 0 <= x < res and 0 <= y < res:
            draw.text((x+12, y-15), str(i), fill=tuple(colors[i]), font=font, 
                     stroke_width=1, stroke_fill=(255, 255, 255))
    
    return np.array(img)
