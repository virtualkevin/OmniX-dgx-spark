import os
import numpy as np
import torch

def export_pointcloud_trajectory(trajectory, colors, base_path, max_points_per_frame=5000, seed=None):
    """
    Save point cloud trajectory with both frame and view organization
    
    Parameters
    ----------
    trajectory : torch.Tensor or np.ndarray
        Shape [im, t, h, w, 3] - point cloud trajectory for multiple views
    colors : torch.Tensor or np.ndarray  
        Shape [im, h, w, 3] - RGB colors for each view (constant across time)
    base_path : str
        Base directory to save all PLY files
    max_points_per_frame : int
        Maximum number of points to save per PLY file (for memory efficiency)
    seed : int
        Random seed for consistent point sampling (default: 42)
    """
    # Use local random state to avoid affecting global random seed
    if seed is None:
        seed = 42
    local_rng = np.random.RandomState(seed)
        
    # Convert to numpy if needed
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    
    # Get dimensions
    n_views, n_frames, height, width = trajectory.shape[:4]
    total_pixels = height * width
    
    # Create base directory
    os.makedirs(base_path, exist_ok=True)
    
    print(f"Saving trajectory: {n_views} views × {n_frames} frames")
    print(f"Output directory: {base_path}")
    
    # For each view, determine which pixels to sample (consistent across all frames)
    for view_idx in range(n_views):
        # Check validity across ALL frames for this view
        view_trajectory = trajectory[view_idx]  # [t, h, w, 3]
        view_colors = colors[view_idx]          # [h, w, 3]
        
        # Find pixels that are valid across ALL frames
        valid_across_time = np.ones((height, width), dtype=bool)
        for frame_idx in range(n_frames):
            frame_points = view_trajectory[frame_idx]  # [h, w, 3]
            frame_valid = np.isfinite(frame_points).all(axis=2) & (np.linalg.norm(frame_points, axis=2) > 1e-6)
            valid_across_time = valid_across_time & frame_valid
        
        # Get valid pixel indices
        valid_h_indices, valid_w_indices = np.where(valid_across_time)
        n_valid_pixels = len(valid_h_indices)
        
        if n_valid_pixels == 0:
            print(f"Warning: No valid pixels for view {view_idx} across all frames")
            continue
            
        # Sample pixels consistently for this view (same pixels across all frames)
        if n_valid_pixels > max_points_per_frame:
            sample_indices = local_rng.choice(n_valid_pixels, max_points_per_frame, replace=False)
            selected_h = valid_h_indices[sample_indices]
            selected_w = valid_w_indices[sample_indices]
        else:
            selected_h = valid_h_indices
            selected_w = valid_w_indices
        
        # n_selected_pixels = len(selected_h)
        # print(f"View {view_idx}: Selected {n_selected_pixels} pixels from {n_valid_pixels} valid pixels")
        
        # Save each frame for this view using the same sampled pixels
        for frame_idx in range(n_frames):
            # Extract points and colors for selected pixels only
            frame_points = view_trajectory[frame_idx, selected_h, selected_w]  # [n_selected, 3]
            frame_colors = view_colors[selected_h, selected_w]                 # [n_selected, 3]
            
            # Convert colors to uint8 [0-255]
            rgb_uint8 = (np.clip(frame_colors, 0, 1) * 255).round().astype(np.uint8)
            
            # Generate filename
            filename = f"image_{view_idx:02d}_time_{frame_idx:04d}.ply"
            filepath = os.path.join(base_path, filename)
            
            # Write PLY file
            _write_ply_file(filepath, frame_points, rgb_uint8)
        
        # print(f"Completed view {view_idx + 1}/{n_views}")
    
    # print(f"Successfully saved {n_views * n_frames} PLY files to {base_path}")


def _write_ply_file(filepath, points, colors):
    """
    Write a single PLY file with points and colors
    
    Parameters
    ----------
    filepath : str
        Output file path
    points : np.ndarray
        Shape [N, 3] - 3D coordinates
    colors : np.ndarray
        Shape [N, 3] - RGB colors as uint8 [0-255]
    """
    n_points = len(points)
    
    # PLY header
    header_lines = [
        "ply",
        "format ascii 1.0", 
        f"element vertex {n_points}",
        "property float x",
        "property float y", 
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header"
    ]
    header = "\n".join(header_lines) + "\n"
    
    # Write file
    with open(filepath, "w") as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")