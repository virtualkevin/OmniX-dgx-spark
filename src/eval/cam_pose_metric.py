# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This script is adapted from PoseDiffusion: https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/metric.py
# The only change is that the original script assumes PyTorch3D SE3 matrices, which is row-major, which assumes the translation vector is at the last row,
# while the script here assumes the standard column-major SE3 matrices that has the translation vector in the last column.

import numpy as np
import torch

from depth_anything_3.utils.geometry import mat_to_quat



def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2

def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg

def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """
    # 检查输入是否包含NaN值
    if torch.isnan(r_error).any() or torch.isnan(t_error).any():
        print("Warning: Input contains NaN values. Filtering them out.")
        valid_mask = ~(torch.isnan(r_error) | torch.isnan(t_error))
        r_error = r_error[valid_mask]
        t_error = t_error[valid_mask]
    
    # 确保输入不为空
    if r_error.numel() == 0 or t_error.numel() == 0:
        print("Error: Empty input tensors after filtering NaN values.")
        return torch.tensor(0.0)  # 返回0作为默认值
    
    # 裁剪极端值
    r_error = torch.clamp(r_error, 0, max_threshold)
    t_error = torch.clamp(t_error, 0, max_threshold)

    # 连接误差张量
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # 计算每对的最大误差值
    max_errors, _ = torch.max(error_matrix, dim=1)
    
    # 使用自定义方法计算直方图，避免torch.histc的潜在问题
    bins = torch.linspace(0, max_threshold, max_threshold + 1)
    histogram = torch.zeros(max_threshold + 1, device=r_error.device)
    
    for i in range(len(bins)-1):
        histogram[i] = ((max_errors >= bins[i]) & (max_errors < bins[i+1])).sum()
    # 最后一个bin包含等于max_threshold的值
    histogram[-1] = (max_errors == max_threshold).sum()
    
    # 归一化直方图
    num_pairs = float(max_errors.size(0))
    if num_pairs > 0:  # 避免除零错误
        normalized_histogram = histogram / num_pairs
    else:
        print("Warning: No valid pairs for histogram calculation.")
        return torch.tensor(0.0)
    
    # 计算累积和
    cumsum = torch.cumsum(normalized_histogram, dim=0)
    
    # 检查结果是否包含NaN
    if torch.isnan(cumsum).any():
        print("Warning: Cumulative sum contains NaN values.")
        return torch.tensor(0.0)
    
    # 返回平均AUC
    return cumsum.mean()

def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE(3) matrix in the batch.

    Args:
        se3 (Tensor): Nx4x4 tensor of SE(3) matrices.

    Returns:
        Tensor: Nx4x4 tensor of inverted SE(3) matrices.
    """
    # Extract rotation matrix R and translation vector t
    R = se3[:, :3, :3]            # Shape: (N, 3, 3)
    t = se3[:, :3, 3].unsqueeze(2)  # Shape: (N, 3, 1)
    # Compute the transpose (inverse) of the rotation matrix
    R_transposed = R.transpose(1, 2)  # Shape: (N, 3, 3)
    # Compute the new translation vector: -R^T * t
    t_inv = -torch.bmm(R_transposed, t)  # Shape: (N, 3, 1)
    # Construct the inverse SE(3) matrix
    inv_se3 = torch.zeros_like(se3)  # Initialize an empty tensor with the same shape
    # Set the rotation part
    inv_se3[:, :3, :3] = R_transposed
    # Set the translation part
    inv_se3[:, :3, 3] = t_inv.squeeze(2)
    # Set the bottom row to [0, 0, 0, 1]
    inv_se3[:, 3, 3] = 1.0

    return inv_se3
