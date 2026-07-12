"""
Simple visualization script for OmniX 4D trajectory prediction.

Reads images from a folder, sorts them, center-crops to a target resolution
(default 280x504, HxW), runs the model, and saves a 3D-trajectory visualization.

Checkpoint and output paths are passed as Hydra overrides, e.g.:

    python visualize_simple.py \
        +experiment=<exp_name> \
        +paths.image_folder=/path/to/images \
        +paths.checkpoint_path=/path/to/model.ckpt \
        +paths.output_path=/path/to/output
"""

import os
from typing import Optional
from pathlib import Path

import hydra
import numpy as np
import cv2
import torch
import PIL.Image
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from omegaconf import DictConfig
from einops import rearrange
from lightning import LightningModule

import rootutils

# Setup project root for imports and env variables
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import sys

ROOT = os.environ["PROJECT_ROOT"]
deformable_detr_path = os.path.join(ROOT, "dependencies", "Deformable_DETR")
if deformable_detr_path not in sys.path:
    sys.path.insert(0, deformable_detr_path)

try:
    LANCZOS = PIL.Image.Resampling.LANCZOS
    BICUBIC = PIL.Image.Resampling.BICUBIC
except AttributeError:
    LANCZOS = PIL.Image.LANCZOS
    BICUBIC = PIL.Image.BICUBIC

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================================
# Image loading & cropping
# ==========================================

def crop_to_resolution(img_pil, target_wh):
    """Rescale (preserving aspect ratio) then center-crop to target (W, H).

    Mirrors the preprocessing in OMNIX/preprocess/generate_dataset.py.
    """
    target_w, target_h = target_wh
    input_w, input_h = img_pil.size

    # Scale up so the image fully covers the target, then center crop.
    scale = max(target_w / input_w, target_h / input_h) + 1e-8
    resize_w = int(np.floor(input_w * scale))
    resize_h = int(np.floor(input_h * scale))
    resample = LANCZOS if scale < 1 else BICUBIC
    img_pil = img_pil.resize((resize_w, resize_h), resample=resample)

    left = (resize_w - target_w) / 2
    top = (resize_h - target_h) / 2
    img_pil = img_pil.crop((left, top, left + target_w, top + target_h))
    return img_pil


def load_images_from_folder(image_folder, target_wh, device="cuda"):
    """Read & sort images, crop to target, and build per-video image info.

    Supports two layouts:
      1. Multi-video: `image_folder` contains subfolders (video_00, video_01, ...),
         each holding the frames of one video.
      2. Single-video: `image_folder` directly contains image files.

    Returns images [N, 3, H, W] and image_info [N, 4] where each row is
    [image_idx, video_idx, local_time_idx, global_time_idx]:
      - image_idx / global_time_idx: position in the flat concatenated sequence
      - video_idx: which video the frame belongs to
      - local_time_idx: position within that video
    """
    folder = Path(image_folder)
    subdirs = sorted(p for p in folder.iterdir() if p.is_dir())

    if subdirs:
        video_folders = subdirs
        print(f"Found {len(video_folders)} video folders in {image_folder}")
    else:
        video_folders = [folder]  # single-video: frames sit directly in folder

    images = []
    video_idx_list, local_time_idx_list = [], []
    for video_idx, vdir in enumerate(video_folders):
        frame_paths = sorted(
            p for p in vdir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not frame_paths:
            continue
        print(f"  video {video_idx} ({vdir.name}): {len(frame_paths)} frames")
        for local_t, path in enumerate(frame_paths):
            img_pil = PIL.Image.open(path).convert("RGB")
            img_pil = crop_to_resolution(img_pil, target_wh)
            images.append(np.asarray(img_pil))
            video_idx_list.append(video_idx)
            local_time_idx_list.append(local_t)

    if not images:
        raise FileNotFoundError(f"No images found in {image_folder}")

    images = np.stack(images) / 255.0  # [N, H, W, 3]
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # [N, 3, H, W]

    n = len(images)
    global_idx = np.arange(n)
    image_info = np.stack([
        global_idx,                       # image_idx
        np.array(video_idx_list),         # video_idx
        np.array(local_time_idx_list),    # local_time_idx
        global_idx,                       # global_time_idx
    ], axis=1)

    images = images.to(device)
    image_info = torch.from_numpy(image_info).to(device)
    return images, image_info


# ==========================================
# Camera helpers
# ==========================================

def move_cameras_backwards(c2ws, distance=0.5, lift_height=0.0, tilt_down_deg=0.0):
    """Push cameras back along the optical axis, lift, and tilt down."""
    new_c2ws = c2ws.copy()
    theta = np.radians(tilt_down_deg)
    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    for i in range(len(new_c2ws)):
        forward_vector = new_c2ws[i, :3, 2]
        new_c2ws[i, :3, 3] -= distance * forward_vector
        new_c2ws[i, 1, 3] -= lift_height
        new_c2ws[i, :3, :3] = new_c2ws[i, :3, :3] @ R_tilt
    return new_c2ws


def ensure_4x4(c2w):
    """Convert (3, 4) or (T, 3, 4) camera poses to homogeneous 4x4 form."""
    if c2w.ndim == 2 and c2w.shape == (3, 4):
        row = np.array([[0, 0, 0, 1]], dtype=c2w.dtype)
        return np.concatenate([c2w, row], axis=0)
    if c2w.ndim == 3 and c2w.shape[1:] == (3, 4):
        row = np.tile(np.array([0, 0, 0, 1], dtype=c2w.dtype), (c2w.shape[0], 1, 1))
        return np.concatenate([c2w, row], axis=1)
    return c2w


# ==========================================
# Saving helpers
# ==========================================

def save_images_nhwc(image_array, save_dir, prefix="cam"):
    """Save an [N, H, W, 3] float/uint8 array as individual jpg images."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(image_array)} GT images to: {save_path}")
    for n, img in enumerate(image_array):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path / f"{prefix}_{n:02d}.jpg"), img_bgr)


# ==========================================
# 3D trajectory visualization
# ==========================================

def visualize_3d_trajectories(
    trajectories,
    intrinsics,
    c2ws,
    foreground_masks,
    image_array,
    confidence_maps=None,
    save_dir="./output_videos",
    num_points_to_show=30,
    traj_length=15,
    traj_alpha=0.8,
    traj_width=2,
    img_h=512,
    img_w=512,
    fps=10,
    show_dense_point_cloud=True,
    project_all_views=False,
    point_size=3,
    orbit_frames=30,
    orbit_radius=0.3,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    traj_width = int(max(1, traj_width))
    N, T, H, W, _ = trajectories.shape

    image_array_uint8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)

    cmap = plt.get_cmap("hsv")
    default_colors = [
        (np.array(cmap(i / max(1, num_points_to_show))[:3]) * 255).astype(int).tolist()
        for i in range(num_points_to_show)
    ]

    traj_trans = trajectories.transpose(0, 2, 3, 1, 4)  # [N, H, W, T, xyz]

    for n in range(N):
        video_path = save_path / f"video_cam_{n:02d}.mp4"
        print(f"Processing Camera {n} -> {video_path}")

        # --- Gather dense point cloud (per-view or all views) ---
        if project_all_views:
            full_c, full_h, full_w = np.meshgrid(
                np.arange(N), np.arange(H), np.arange(W), indexing="ij")
            full_c, full_h, full_w = full_c.flatten(), full_h.flatten(), full_w.flatten()
            flat_conf = confidence_maps.flatten() if confidence_maps is not None else None
        else:
            full_c = np.full(H * W, n)
            grid_h, grid_w = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            full_h, full_w = grid_h.flatten(), grid_w.flatten()
            flat_conf = confidence_maps[n].flatten() if confidence_maps is not None else None

        if flat_conf is not None:
            threshold = np.percentile(flat_conf, 1)
            valid_mask = flat_conf > threshold
            full_c, full_h, full_w = full_c[valid_mask], full_h[valid_mask], full_w[valid_mask]
        else:
            threshold = -1.0

        cloud_pts_3d = traj_trans[full_c, full_h, full_w]
        cloud_colors = image_array_uint8[full_c, full_h, full_w]

        # --- Sample foreground trajectory points to draw as tracks ---
        if project_all_views:
            fg_mask_bool = foreground_masks > 0.1
        else:
            fg_mask_bool = np.zeros_like(foreground_masks, dtype=bool)
            fg_mask_bool[n] = foreground_masks[n] > 0.5
        if confidence_maps is not None:
            fg_mask_bool = fg_mask_bool & (confidence_maps > threshold)

        fg_indices = np.argwhere(fg_mask_bool)
        if len(fg_indices) > 0:
            f_c, f_h, f_w = fg_indices[:, 0], fg_indices[:, 1], fg_indices[:, 2]
            if confidence_maps is not None:
                conf_vals = confidence_maps[f_c, f_h, f_w]
                p = conf_vals / (conf_vals.sum() + 1e-8)
                s_idx = np.random.choice(
                    len(fg_indices), size=min(num_points_to_show, len(fg_indices)),
                    replace=False, p=p)
            else:
                s_idx = np.random.choice(
                    len(fg_indices), size=min(num_points_to_show, len(fg_indices)),
                    replace=False)
            sampled_pts_3d = traj_trans[f_c[s_idx], f_h[s_idx], f_w[s_idx]]
            sort_order = np.argsort(sampled_pts_3d[:, 0, 1])  # sort by y
            sampled_pts_3d = sampled_pts_3d[sort_order]
            sampled_colors = default_colors[:len(s_idx)]
        else:
            sampled_pts_3d = []

        # --- Camera + intrinsics scaled to render resolution ---
        c2w_orig = c2ws[n]
        K = intrinsics[n].copy()
        K[0, 0] *= img_w / W
        K[1, 1] *= img_h / H
        K[0, 2] *= img_w / W
        K[1, 2] *= img_h / H

        frames_list = []

        def render_frame(c2w_render, t, traj_t_start, traj_t_end, z_near):
            """Render one frame: dense cloud + sampled tracks for [traj_t_start, traj_t_end)."""
            w2c = np.linalg.inv(c2w_render)
            R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]
            frame = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

            if show_dense_point_cloud:
                P_c = (R_w2c @ cloud_pts_3d[:, t, :].T).T + t_w2c
                z_c = P_c[:, 2]
                valid_z = z_c > z_near
                if np.any(valid_z):
                    P_c_v, C_v, Z_v = P_c[valid_z], cloud_colors[valid_z], z_c[valid_z]
                    uv_h = (K @ P_c_v.T).T
                    u = np.round(uv_h[:, 0] / (uv_h[:, 2] + 1e-8)).astype(int)
                    v = np.round(uv_h[:, 1] / (uv_h[:, 2] + 1e-8)).astype(int)
                    in_b = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                    u_s, v_s, c_s, z_s = u[in_b], v[in_b], C_v[in_b], Z_v[in_b]
                    sort_idx = np.argsort(-z_s)  # paint far-to-near
                    u_s, v_s, c_s = u_s[sort_idx], v_s[sort_idx], c_s[sort_idx]
                    offset = -(point_size // 2)
                    for dy in range(offset, offset + point_size):
                        for dx in range(offset, offset + point_size):
                            vv = np.clip(v_s + dy, 0, img_h - 1)
                            uu = np.clip(u_s + dx, 0, img_w - 1)
                            frame[vv, uu] = c_s

            if len(sampled_pts_3d) > 0:
                overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                t_win = traj_t_end - traj_t_start
                curr_traj_3d = sampled_pts_3d[:, traj_t_start:traj_t_end, :]
                for i in range(len(sampled_pts_3d)):
                    color_rgb = sampled_colors[i]
                    track_c = (R_w2c @ curr_traj_3d[i].T).T + t_w2c
                    uv_h = (K @ track_c.T).T
                    pts_2d = np.round(uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)).astype(np.int32)
                    v_z = track_c[:, 2] > 0.3
                    for k in range(t_win - 1):
                        if not (v_z[k] and v_z[k + 1]):
                            continue
                        alpha = int(255 * traj_alpha * (0.5 + 0.5 * (k + 1) / t_win))
                        cv2.line(overlay, tuple(pts_2d[k]), tuple(pts_2d[k + 1]),
                                 (color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                                 traj_width, cv2.LINE_AA)
                    if v_z[-1]:
                        cv2.circle(overlay, tuple(pts_2d[-1]), traj_width,
                                   (color_rgb[0], color_rgb[1], color_rgb[2],
                                    int(255 * traj_alpha)), -1)
                mask_a = overlay[:, :, 3:] / 255.0
                frame = (overlay[:, :, :3] * mask_a + frame * (1 - mask_a)).astype(np.uint8)
            return frame

        # Normal time-stepping
        for t in range(T):
            t_start = max(0, t - traj_length + 1)
            frames_list.append(render_frame(c2w_orig, t, t_start, t + 1, z_near=0.001))

        # Orbit around the final frame
        t_final = T - 1
        cam_pos = c2w_orig[:3, 3]
        cam_up = c2w_orig[:3, 1]
        cam_right = np.cross(c2w_orig[:3, 2], cam_up)
        cam_right /= np.linalg.norm(cam_right) + 1e-8
        for orbit_i in range(orbit_frames):
            angle = 2 * np.pi * orbit_i / orbit_frames
            offset = orbit_radius * (np.cos(angle) * cam_right + np.sin(angle) * cam_up)
            c2w_orbit = c2w_orig.copy()
            c2w_orbit[:3, 3] = cam_pos + offset
            frames_list.append(render_frame(c2w_orbit, t_final, 0, T, z_near=0.3))

        clip = mpy.ImageSequenceClip(frames_list, fps=fps)
        clip.write_videofile(
            str(video_path),
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            audio=False,
            logger=None,
        )


# ==========================================
# Main
# ==========================================

def run(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")

    image_folder = cfg.paths.image_folder
    checkpoint_path = cfg.paths.checkpoint_path
    output_path = cfg.paths.output_path
    target_h = cfg.paths.get("target_h", 280)
    target_w = cfg.paths.get("target_w", 504)
    os.makedirs(output_path, exist_ok=True)

    # Instantiate & load model
    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.pretrained = checkpoint_path
    model._load_pretrained_weights()
    model = model.to("cuda").eval()

    # Load & crop images
    images, image_info = load_images_from_folder(image_folder, (target_w, target_h))
    batch = {"image": images.unsqueeze(0), "image_info": image_info.unsqueeze(0)}

    with torch.no_grad():
        preds = model(batch)

    trajectory = preds["trajectory"][0].cpu().numpy()
    c2w = ensure_4x4(preds["camera_pose"][0].cpu().numpy())
    intrinsic = preds["intrinsics"][0].cpu().numpy()

    images_gt = batch["image"][0].cpu().numpy()
    images_hwc = rearrange(images_gt, "im c h w -> im h w c")
    im, img_h, img_w = images_hwc.shape[:3]

    c2w = move_cameras_backwards(c2w, distance=0.16, lift_height=0.1, tilt_down_deg=-6)
    foreground_masks = preds["pts3d_dynamic_score"][0].cpu().numpy()
    num_points_to_show = max(1, int(((foreground_masks > 0.5).sum() // im) * 0.03))

    print(f"Saving visualization to {output_path}")
    visualize_3d_trajectories(
        trajectories=trajectory,
        intrinsics=intrinsic,
        c2ws=c2w,
        foreground_masks=foreground_masks,
        image_array=images_hwc,
        confidence_maps=None,
        save_dir=output_path,
        num_points_to_show=num_points_to_show,
        traj_length=im,
        traj_alpha=1.0,
        traj_width=3,
        img_h=img_h * 4,
        img_w=img_w * 4,
        fps=15,
        show_dense_point_cloud=True,
        project_all_views=False,
        point_size=6,
        orbit_radius=0.1,
    )

    save_images_nhwc(images_hwc, os.path.join(output_path, "gt_images"))


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    run(cfg)
    return None


if __name__ == "__main__":
    main()
