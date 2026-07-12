import numpy as np
import torch

### user func
def camera_to_pixel_coordinates(points_cam, camera_intrinsic, image_shape, return_depth=False, has_batch=False):
    """
    Project 3D points from camera coordinates to pixel coordinates.

    Args:
        points_cam: (..., 3) NumPy or PyTorch array - points in camera coordinates
        camera_intrinsic: (..., 3, 3) matching type and batch layout - camera intrinsics
        image_shape: (H, W) - image dimensions
        return_depth: If True, also return depth map from camera coords

    Returns:
        uv: (..., 2) - pixel coordinates (float)
        valid_mask: (...) boolean mask, True if point is in front of camera and within frame
        depth: (...) depth values (only if return_depth=True)
    """
    # Align shapes for broadcasting
    pts_al, K_al = align_dims_for_broadcast(points_cam, camera_intrinsic, has_batch=has_batch)  # (..., 3) and (..., 3, 3)

    # Apply intrinsic matrix: u,v,w = K * [x, y, z]^T
    # First make points homogeneous 3D column format
    pts_col = pts_al[..., None]  # (..., 3, 1)
    uvw = K_al @ pts_col  # (..., 3, 1)
    uvw = uvw.squeeze(-1)  # (..., 3)

    # Perspective division
    u = uvw[..., 0] / uvw[..., 2]
    v = uvw[..., 1] / uvw[..., 2]
    depth = uvw[..., 2]  # This ensures correct shape (same as u/v)

    # Build uv coords
    if isinstance(points_cam, np.ndarray):
        uv = np.stack([u, v], axis=-1)
    else:
        uv = torch.stack([u, v], dim=-1)

    # Validity mask
    H, W = image_shape
    # opencv
    within_x = (uv[..., 0] >= -0.5) & (uv[..., 0] < W - 0.5)
    within_y = (uv[..., 1] >= -0.5) & (uv[..., 1] < H - 0.5)
    valid_depth = depth > 0
    valid_mask = within_x & within_y & valid_depth

    if return_depth:
        return uv, valid_mask, depth
    else:
        return uv, valid_mask


def world_to_pixel_coordinates(points_world, camera_intrinsic, camera_pose, image_shape, return_depth=False, has_batch=False):
    """
    Project 3D points from world coordinates to pixel coordinates.
    """
    # Transform world points to camera coords
    points_cam = world_to_camera_coordinates(points_world, camera_pose, has_batch=has_batch)
    return camera_to_pixel_coordinates(points_cam, camera_intrinsic, image_shape, return_depth=return_depth, has_batch=has_batch)


def camera_to_world_coordinates(points_cam, camera_pose, has_batch=False):
    """
    Transform points from camera coordinates to world coordinates.
    """
    pts_al, pose_al = align_dims_for_broadcast(points_cam, camera_pose, has_batch=has_batch)
    return transform_3d(pts_al, pose_al)


def world_to_camera_coordinates(points_world, camera_pose, has_batch=False):
    """
    Transform points from world coordinates to camera coordinates.
    """
    w2c = compute_world_to_camera_matrix(camera_pose)
    pts_al, w2c_al = align_dims_for_broadcast(points_world, w2c, has_batch=has_batch)
    return transform_3d(pts_al, w2c_al)


def compute_world_to_camera_matrix(camera_pose):
    """
    Compute world-to-camera transformation from camera-to-world pose.
    """
    return closed_form_inverse_se3(camera_pose)

def depthmap_to_camera_coordinates(depthmap, camera_intrinsic, return_valid_mask=False, has_batch=False):
    """
    Convert depthmap to camera coordinates.

    Args:
        depthmap: (..., H, W) - numpy array or torch tensor
        camera_intrinsic: (..., 3, 3) - camera intrinsic matrix
        return_valid_mask: bool - whether to return valid mask
        has_batch: bool - if True, first batch dim is paired; if False, Cartesian product mode

    Returns:
        camera_coordinates: (..., H, W, 3) - camera coordinates
        valid_mask: (..., H, W) - optional valid mask where depth > 0
    """
    # Detect input type
    if hasattr(depthmap, 'device'):  # torch tensor
        lib = torch
        device = depthmap.device
        dtype = depthmap.dtype
    else:  # numpy array
        lib = np
        device = None
        dtype = depthmap.dtype

    # Get dimensions
    *batch_dims, H, W = depthmap.shape
    depth_batch_dims = len(batch_dims)
    intrinsic_batch_dims = camera_intrinsic.ndim - 2  # exclude last (3, 3)

    # Align dimensions using slice approach similar to align_dims_for_broadcast
    skip_dims = 0
    if has_batch:
        if depth_batch_dims == 0 or intrinsic_batch_dims == 0:
            raise ValueError("has_batch=True but one input has no batch dims")
        if batch_dims[0] != camera_intrinsic.shape[0]:
            raise ValueError(f"has_batch=True but first dim differ: depthmap({batch_dims[0]}) vs intrinsic({camera_intrinsic.shape[0]})")
        skip_dims = 1

    # Calculate remaining batch dims to expand
    depth_rest = depth_batch_dims - skip_dims
    intrinsic_rest = intrinsic_batch_dims - skip_dims

    # Align camera_intrinsic using slice expansion
    # Keep `skip_dims` leading dims, insert None for remaining depth batch dims, keep coordinate dims
    intrinsic_slice = (slice(None),) * skip_dims + \
                     (slice(None),) * intrinsic_rest + \
                     (None,) * depth_rest + \
                     (slice(None), slice(None))
    K_aligned = camera_intrinsic[intrinsic_slice]

    # Extract intrinsic parameters
    fu = K_aligned[..., 0, 0]  # (..., H, W) after broadcasting
    fv = K_aligned[..., 1, 1]
    cu = K_aligned[..., 0, 2]
    cv = K_aligned[..., 1, 2]

    # Create pixel coordinate grids
    if device is not None:
        u, v = lib.meshgrid(lib.arange(W, device=device), lib.arange(H, device=device), indexing='xy')
    else:
        u, v = lib.meshgrid(lib.arange(W), lib.arange(H), indexing='xy')

    # Convert to correct dtype and expand to match depthmap batch dims
    u = u.astype(dtype) if device is None else u.to(dtype)
    v = v.astype(dtype) if device is None else v.to(dtype)

    # Add batch dimensions to u, v to match depthmap
    uv_slice = (None,) * depth_batch_dims + (slice(None), slice(None))
    u = u[uv_slice]
    v = v[uv_slice]

    # Add spatial dimensions to intrinsic parameters for broadcasting
    intrinsic_spatial_slice = (..., None, None)
    fu = fu[intrinsic_spatial_slice]
    fv = fv[intrinsic_spatial_slice]
    cu = cu[intrinsic_spatial_slice]
    cv = cv[intrinsic_spatial_slice]

    # Convert to camera coordinates
    z_cam = depthmap  # (..., H, W)
    x_cam = (u - cu) * z_cam / fu  # (..., H, W)
    y_cam = (v - cv) * z_cam / fv  # (..., H, W)

    # Stack into 3D coordinates
    if device is not None:
        camera_coords = lib.stack([x_cam, y_cam, z_cam], dim=-1)  # (..., H, W, 3)
    else:
        camera_coords = lib.stack([x_cam, y_cam, z_cam], axis=-1)  # (..., H, W, 3)

    # Valid mask
    valid_mask = depthmap > 0.0  # (..., H, W)

    if return_valid_mask:
        return camera_coords, valid_mask
    else:
        return camera_coords

def depthmap_to_world_coordinates(depthmap, camera_intrinsic, camera_pose, return_valid_mask=False, has_batch=False):
    """
    Convert depthmap to world coordinates by first converting to camera coords, then to world.

    Args:
        depthmap: (..., H, W) - numpy array or torch tensor
        camera_intrinsic: (..., 3, 3) - camera intrinsic matrix
        camera_pose: (..., 4, 4) - camera pose matrix
        return_valid_mask: bool - whether to return valid mask
        has_batch: bool - if True, first batch dim is paired; if False, Cartesian product mode

    Returns:
        world_coordinates: (..., H, W, 3) - world coordinates
        valid_mask: (..., H, W) - optional valid mask where depth > 0
    """
    # Step 1: Convert depthmap to camera coordinates
    if return_valid_mask:
        camera_coords, valid_mask = depthmap_to_camera_coordinates(
            depthmap, camera_intrinsic, return_valid_mask=True, has_batch=has_batch
        )
    else:
        camera_coords = depthmap_to_camera_coordinates(
            depthmap, camera_intrinsic, return_valid_mask=False, has_batch=has_batch
        )

    # Step 2: Transform to world coordinates directly
    # camera_to_world_coordinates can handle (..., H, W, 3) -> (..., H, W, 3) naturally!
    world_coords = camera_to_world_coordinates(camera_coords, camera_pose, has_batch=has_batch)

    if return_valid_mask:
        return world_coords, valid_mask
    else:
        return world_coords
    
### basic ops
def align_dims_for_broadcast(points, cameras, has_batch=False):
    """
    Align points and cameras for broadcasting.

    Args:
        points: [..., 3] ndarray or Tensor
        cameras: [..., c1, c2] ndarray or Tensor
        has_batch (bool):
            - False: Cartesian product mode — all batch dims independent.
            - True: Paired-batch mode — first batch dim matches, not expanded, others are broadcast.

    Returns:
        points_aligned, cameras_aligned
    """
    # How many batch dims (excluding coordinate dims)
    pb_dims = points.ndim - 1
    cb_dims = cameras.ndim - 2

    # Check first dim if paired-batch mode
    skip_dims = 0
    if has_batch:
        if points.shape[0] != cameras.shape[0]:
            raise ValueError(f"has_batch=True but first dim differ: {points.shape[0]} vs {cameras.shape[0]}")
        skip_dims = 1  # skip expansion on the first batch dim

    # Number of batch dims to expand (excluding skipped first dim)
    pb_rest = pb_dims - skip_dims
    cb_rest = cb_dims - skip_dims

    # Slice for points:
    #  - keep `skip_dims` leading dims
    #  - insert None for each remaining camera batch dim
    #  - then keep original point dims
    points_slice = (slice(None),) * skip_dims + (None,) * cb_rest + (slice(None),) * (pb_dims - skip_dims + 1)
    points_aligned = points[points_slice]

    # Slice for cameras:
    #  - keep `skip_dims` leading dims
    #  - keep remaining camera batch dims
    #  - insert None for each remaining point batch dim
    #  - keep last 2 coordinate dims
    cameras_slice = (slice(None),) * skip_dims + (slice(None),) * cb_rest + (None,) * pb_rest + (slice(None), slice(None))
    cameras_aligned = cameras[cameras_slice]

    return points_aligned, cameras_aligned

def transform_3d(points, transforms):
    """
    Apply SE(3) transformations to 3D points.

    Args:
        points: (..., 3) ndarray or Tensor
        transforms: (..., 4, 4) ndarray or Tensor

    Returns:
        transformed_points: (..., 3)
    """
    # Extract rotation and translation from transform
    R = transforms[..., :3, :3]           # (..., 3, 3)
    t = transforms[..., :3, 3]            # (..., 3)

    # Apply transformation: R @ p + t
    # Use broadcasting: points shape (..., 3), R shape (..., 3, 3)
    transformed_points = (R @ points[..., None])[..., 0] + t

    return transformed_points


def closed_form_inverse_se3(se3, return_4x4=True):
    """
    Compute the inverse of batched SE(3) transformation matrices.

    Args:
        se3: (..., 3, 4) | (..., 4, 4) NumPy ndarray or PyTorch tensor
             SE(3) matrices with last two dims 4x4.
             Works for single matrix or any number of batch dimensions.

    Returns:
        inverses: (..., 4, 4) same type and shape as input.
    """
    if se3.shape[-2:] not in [(3, 4), (4, 4)]:
        raise ValueError(f"Expected shape (..., 3, 4) or (..., 4, 4), got {se3.shape}")

    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] == (3, 4):
        if is_numpy:
            ones_row = np.zeros((*se3.shape[:-2], 1, 4), dtype=se3.dtype)
            ones_row[..., 0, 3] = 1
            se3 = np.concatenate([se3, ones_row], axis=-2)
        else:
            ones_row = torch.zeros((*se3.shape[:-2], 1, 4), dtype=se3.dtype, device=se3.device)
            ones_row[..., 0, 3] = 1
            se3 = torch.cat([se3, ones_row], dim=-2)

    # Extract rotation R and translation t
    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]  # keep as column vector

    # Transpose rotation matrix
    RT = R.transpose(-1, -2) if not is_numpy else np.swapaxes(R, -1, -2)

    # Compute -R^T * t
    top_right = -(RT @ t)

    # Create identity matrix template matching backend, dtype, and device
    if is_numpy:
        eye = np.eye(4, dtype=se3.dtype)
        # np.broadcast_to results in a read-only view, so copy for assignment
        inv_se3 = np.broadcast_to(eye, se3.shape).copy()
    else:
        eye = torch.eye(4, dtype=se3.dtype, device=se3.device)
        # Use .clone() to allow writing into expanded tensor
        inv_se3 = eye.expand(se3.shape).clone()

    # Fill in rotation and translation components
    inv_se3[..., :3, :3] = RT
    inv_se3[..., :3, 3:4] = top_right

    # Ensure bottom row is [0, 0, 0, 1]
    inv_se3[..., 3, :] = 0
    inv_se3[..., 3, 3] = 1

    if return_4x4:
        return inv_se3
    else:
        return inv_se3[..., :3, :]
