import torch

def draw_points(image, points, color, radius=2.5, antialias=False, device=None):
    """
    Draw colored points on an image (H, W, 3) given 'points' (N, 2) in UV coords.
    Filters out points outside image bounds. Supports antialiasing.
    """
    if device is None:
        device = image.device
    img = image.clone().to(device)
    points = points.to(device)
    H, W, C = img.shape
    N = points.shape[0]

    # Normalize color to shape (N, 3)
    if isinstance(color, (list, tuple)) and len(color) == 3:
        color = torch.tensor(color, dtype=torch.float32, device=device).unsqueeze(0).repeat(N, 1)
    elif isinstance(color, torch.Tensor):
        color = color.to(device)
        assert color.shape == (N, 3), "color tensor must be (N, 3)"
    else:
        raise ValueError("color must be (3,) or (N, 3)")

    # Filter invalid points (outside image)
    valid_mask = (points[:, 0] >= 0) & (points[:, 0] < W) & \
                 (points[:, 1] >= 0) & (points[:, 1] < H)
    if valid_mask.sum() == 0:
        return img  # nothing to draw
    points = points[valid_mask]
    color = color[valid_mask]
    N = points.shape[0]

    # Precompute local offsets for a circle radius
    r_floor = int(radius)
    off_y = torch.arange(-r_floor, r_floor + 1, device=device)
    off_x = torch.arange(-r_floor, r_floor + 1, device=device)
    dy, dx = torch.meshgrid(off_y, off_x, indexing="ij")
    dist = torch.sqrt((dx.float())**2 + (dy.float())**2)

    if antialias:
        alpha_offsets = torch.clamp(radius - dist, 0, 1)
        mask_offsets = dist <= radius + 0.5
    else:
        alpha_offsets = (dist <= radius).float()
        mask_offsets = dist <= radius

    offsets = torch.stack([dx[mask_offsets], dy[mask_offsets]], dim=-1)  # (K, 2)
    alpha_offsets = alpha_offsets[mask_offsets]  # (K,)

    # Compute all pixel coords for all points
    all_coords = points[:, None, :] + offsets[None, :, :]
    all_coords = all_coords.round().long()
    px = all_coords[..., 0].view(-1)
    py = all_coords[..., 1].view(-1)
    alpha_all = alpha_offsets.repeat(N)
    colors_all = color[:, None, :].expand(N, alpha_offsets.shape[0], 3).reshape(-1, 3)

    # Filter coords outside image (after offset)
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (alpha_all > 0)
    px, py, alpha_all, colors_all = px[in_bounds], py[in_bounds], alpha_all[in_bounds], colors_all[in_bounds]

    # Blend colors
    img[py, px] = img[py, px] * (1 - alpha_all.unsqueeze(-1)) + colors_all * alpha_all.unsqueeze(-1)

    return img
