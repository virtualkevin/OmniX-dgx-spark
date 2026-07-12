from math import ceil, floor

import torch
import torch.nn.functional as F

from einops import rearrange

from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
from src.utils.projection import closed_form_inverse_se3


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100, label = "", warning=True, mask_tensor=None):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """
        
    if (torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any()) and warning:
        for _ in range(10):
            print(f"{label}, {loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )
    if warning is False and mask_tensor is not None:
        if loss_tensor.ndim == mask_tensor.ndim:
            mask_tensor = torch.where(
                torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
                torch.tensor(False, device=loss_tensor.device),
                mask_tensor
            )
            loss_tensor = torch.where(
                torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
                torch.tensor(0.0, device=loss_tensor.device),
                loss_tensor
            ) 
            
        return loss_tensor, mask_tensor

    if hard_max is not None:
        loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor

def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):

    # List of predicted pose encodings per stage
    pred_pose_encodings = [pred_dict['pose_enc']]
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['valid_mask']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    # TODO: check this, depth-anything use c2w prediction
    gt_extrinsics = batch_data['camera_pose']
    # gt_extrinsics = closed_form_inverse_se3(gt_extrinsics)
    gt_intrinsics = batch_data['intrinsic']
    image_hw = batch_data['image'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, # pose_encoding_type=pose_encoding_type
    )

    # Note: stereo4d intrinsic may be inaccurate, so we replace it with pred_intrinsics
    dataset_names = batch_data["scene_meta"]["dataset_name"]
    if any(x == "stereo4d" for x in dataset_names):
        gt_pose_encoding = gt_pose_encoding.clone()
        gt_pose_encoding[
            torch.tensor([x == "stereo4d" for x in dataset_names],
                        device=gt_pose_encoding.device),
            :, -2:
        ] = pred_dict["pose_enc"][
            torch.tensor([x == "stereo4d" for x in dataset_names],
                        device=pred_dict["pose_enc"].device),
            :, -2:
        ].detach()

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL

def huber_loss(pred, target, beta=1.0):
    """
    计算 Huber 损失。
    
    Huber 损失结合了 L1 和 L2 损失的优点：
    - 当误差小于 beta 时，表现为 L2 损失 (0.5 * error^2)
    - 当误差大于 beta 时，表现为 L1 损失 (beta * (|error| - 0.5 * beta))
    
    这使得它对异常值不那么敏感，同时在误差较小时保持 L2 损失的平滑性质。
    
    Args:
        pred (torch.Tensor): 预测值张量
        target (torch.Tensor): 目标值张量
        beta (float): Huber 损失的阈值参数，控制 L1 和 L2 损失的切换点
        
    Returns:
        torch.Tensor: 计算得到的 Huber 损失
    """
    # 计算预测值与目标值之间的绝对误差
    diff = torch.abs(pred - target)
    
    # 创建掩码，标识误差小于 beta 的元素
    mask = diff < beta
    
    # 对于误差小于 beta 的部分，使用平方损失 (L2)
    squared_loss = 0.5 * diff ** 2
    
    # 对于误差大于等于 beta 的部分，使用线性损失 (L1)，但进行偏移以确保在 beta 处连续
    linear_loss = beta * (diff - 0.5 * beta)
    
    # 根据掩码选择合适的损失类型
    loss = torch.where(mask, squared_loss, linear_loss)
    
    return loss


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, norm="l2", **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth = pred_depth[..., None]
    pred_depth_conf = predictions['depth_conf']

    if "depth" not in batch:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    gt_depth = batch['depth']
    gt_depth_mask = batch['valid_mask'].clone() & torch.isfinite(gt_depth)   # 3D points derived from depth map, so we use the same mask

    gt_depth, gt_depth_mask = check_and_fix_inf_nan(gt_depth, "gt_depth", warning=False, mask_tensor=gt_depth_mask)
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement

    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range, norm=norm)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict

def compute_ray_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, use_conf_loss=False, norm="l2", **kwargs):
    """
    Compute ray loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_ray = predictions['ray']
    pred_ray_conf = predictions['ray_conf']

    h_im, w_im = batch["image"].shape[-2:]
    hr, wr = pred_ray.shape[2:4]
    gt_camera_pose = batch['camera_pose']
    gt_intrinsics = batch["intrinsic"]
    gt_ray = pose_to_ray_map(gt_camera_pose, hr, wr, intrinsics=gt_intrinsics, intrinsics_image_size=(h_im, w_im))
    
    gt_ray_mask = torch.ones_like(pred_ray[..., 0]).bool() & torch.isfinite(pred_ray).any(dim=-1)   # 3D points derived from ray map, so we use the same mask
    gt_ray, gt_ray_mask = check_and_fix_inf_nan(gt_ray, "gt_ray", warning=False, mask_tensor=gt_ray_mask) 
    

    if gt_ray_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_ray).mean()
        loss_dict = {f"loss_conf_ray": dummy_loss,
                    f"loss_reg_ray": dummy_loss,
                    f"loss_grad_ray": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_ray, gt_ray, gt_ray_mask, conf=pred_ray_conf if use_conf_loss else None,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range, norm=norm)

    loss_dict = {
        f"loss_conf_ray": loss_conf,
        f"loss_reg_ray": loss_reg,    
        f"loss_grad_ray": loss_grad,
    }

    return loss_dict

def compute_gsdepth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, norm="l2", **kwargs):
    """
    Compute gsdepth loss.
    
    Args:
        predictions: Dict containing 'gs_depth' and 'gs_depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['gs_depth']
    pred_depth_conf = predictions['gs_depth_conf']

    gt_depth = batch['depth'][:, :pred_depth.shape[1]]
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['valid_mask'].clone()[:, :pred_depth.shape[1]]   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_gsdepth": dummy_loss,
                    f"loss_reg_gsdepth": dummy_loss,
                    f"loss_grad_gsdepth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range, norm=norm)

    loss_dict = {
        f"loss_conf_gsdepth": loss_conf,
        f"loss_reg_gsdepth": loss_reg,    
        f"loss_grad_gsdepth": loss_grad,
    }

    return loss_dict
    

def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, norm="l2", **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['pts3d']
    pred_points_conf = predictions['pts3d_conf']
    if "pts3d" in batch:
        gt_points = batch['pts3d']
    elif "trajectory" in batch:
        gt_traj = batch["trajectory"]
        b, im, t, h, w, _ = gt_traj.shape
        b_idx = torch.arange(b, device=gt_traj.device)[:, None]
        im_idx = torch.arange(im, device=gt_traj.device)[None, :]
        gt_points = gt_traj[b_idx, im_idx, im_idx]
    else:

        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict

    gt_points_mask = batch['valid_mask'] & (torch.isfinite(gt_points).any(dim=-1))
    
    gt_points, gt_point_mask = check_and_fix_inf_nan(gt_points, "gt_points", warning=False, mask_tensor=gt_points_mask)
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range, norm=norm)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_trajectory_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, foreground_weight=1.0, foreground_prob=-1, use_conf_loss=True, norm="l2", **kwargs):
    """
    Compute trajectory loss. Same as compute_point_loss, but for trajectory.
    
    Args:
        predictions: Dict containing 'trajectory' and 'trajectory_conf'
        batch: Dict containing ground truth 'trajectory' and 'valid_masks', not valid mask is only for general use
            and trjectory gt might be nan, which indicates invalid trajectory
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_trajectory = predictions['trajectory']
    if use_conf_loss:
        pred_trajectory_conf = predictions['trajectory_conf']

    if "trajectory" not in batch:
        dummy_loss = (0.0 * pred_trajectory).mean()
        loss_dict = {
            f"loss_conf_trajectory": dummy_loss,
            f"loss_reg_trajectory": dummy_loss,
            f"loss_grad_trajectory": dummy_loss,
        }
        return loss_dict
    gt_trajectory = batch['trajectory']
    gt_valid_mask = batch['valid_mask']
    gt_trajectory_valid_mask = gt_valid_mask[:, :, None] & (torch.isfinite(gt_trajectory).any(dim=-1))
    gt_trajectory_foreground_mask = batch.get('trajectory_foreground', None)

    gt_trajectory, gt_valid_mask = check_and_fix_inf_nan(gt_trajectory, "gt_trajectory", warning=False, mask_tensor=gt_valid_mask) # gt_trajectory has nan is expected
    
    if gt_trajectory_valid_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_trajectory).mean()
        loss_dict = {
            f"loss_conf_trajectory": dummy_loss,
            f"loss_reg_trajectory": dummy_loss,
            f"loss_grad_trajectory": dummy_loss,
        }
        return loss_dict
    
    pred_trajectory_reshaped = rearrange(pred_trajectory, "b im t h w c -> b (im t) h w c")
    gt_trajectory_reshaped = rearrange(gt_trajectory, "b im t h w c -> b (im t) h w c")
    gt_trajectory_valid_mask_reshaped = rearrange(gt_trajectory_valid_mask, "b im t h w -> b (im t) h w")
    if use_conf_loss:
        pred_trajectory_conf_reshaped = rearrange(pred_trajectory_conf, "b im t h w -> b (im t) h w")
    gt_trajectory_foreground_mask_reshaped = rearrange(gt_trajectory_foreground_mask, "b im t h w -> b (im t) h w") if gt_trajectory_foreground_mask is not None else None

    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_trajectory_reshaped, gt_trajectory_reshaped, gt_trajectory_valid_mask_reshaped, conf=pred_trajectory_conf_reshaped if use_conf_loss else None,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range, \
                                                foreground_mask=gt_trajectory_foreground_mask_reshaped, foreground_weight=foreground_weight, foreground_prob=foreground_prob, norm=norm)
    
    loss_dict = {
        f"loss_conf_trajectory": loss_conf,
        f"loss_reg_trajectory": loss_reg,
        f"loss_grad_trajectory": loss_grad,
    }
    
    return loss_dict

# regression loss with wrapper
def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1, foreground_mask=None, foreground_weight=1.0, foreground_prob=-1, norm="l1"):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth pointsß
    if norm == "l1":
        loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1, p=1)
    elif norm == "l2":
        loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    else:
        loss_reg = F.smooth_l1_loss(gt[mask], pred[mask], reduction="none").sum(dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    if conf is not None:
        loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
    else:
        loss_conf = 0 * loss_reg
        
    # Initialize gradient loss
    loss_grad = 0

    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )
    
    # Use foreground weight if provided
    foreground_weights = torch.ones(bb, ss, hh, ww, device=pred.device)
    if foreground_prob == -1 or foreground_mask is None:
        if foreground_mask is not None and foreground_weight != 1.0:
            foreground_weights = torch.ones(bb, ss, hh, ww, device=pred.device)
            foreground_weights[foreground_mask] = foreground_weight
            loss_conf = loss_conf * foreground_weights[mask]
            loss_reg = loss_reg * foreground_weights[mask]
    else:
        assert foreground_mask is not None, "foreground_mask cannot be None when foreground_prob != -1"
        # import ipdb
        # ipdb.set_trace()
        # 1. 初始化权重矩阵 (默认为 0，即不参与计算)
        foreground_weights = torch.zeros(bb, ss, hh, ww, device=pred.device)
        
        # 2. 结合有效性 mask 和 前景 mask
        valid_fg_mask = (foreground_mask > 0) & mask
        valid_bg_mask = (foreground_mask == 0) & mask
        
        num_fg = valid_fg_mask.sum()
        num_bg = valid_bg_mask.sum()
        
        # 3. 执行采样逻辑
        if num_fg > 100:
            # --- 步骤 A: 激活所有前景点 ---
            foreground_weights[valid_fg_mask] = 1.0
            
            # --- 步骤 B: 根据比例计算需要采样的背景点数量 ---
            # 防止除以 0
            safe_prob = max(foreground_prob, 1e-6)
            
            if safe_prob >= 1.0:
                # 如果 prob 是 1.0，只要前景，不要背景
                num_bg_to_sample = 0
            else:
                num_bg_to_sample = int(num_fg * (1.0 - safe_prob) / safe_prob)
            
            # 确保不也超过实际拥有的背景点数
            num_bg_to_sample = min(num_bg_to_sample, num_bg)
            
            if num_bg_to_sample > 0:
                # --- 步骤 C: 随机抽取背景点 ---
                # 获取所有有效背景点的坐标索引
                # torch.nonzero 返回 [N, 4] 的坐标 (b, s, h, w)
                bg_indices = torch.nonzero(valid_bg_mask, as_tuple=False)
                
                # 随机打乱索引
                perm = torch.randperm(bg_indices.size(0), device=pred.device)
                
                # 取前 num_bg_to_sample 个
                selected_bg_indices = bg_indices[perm[:num_bg_to_sample]]
                
                # 将选中的背景点权重设为 1
                # 使用 split 解包坐标: b, s, h, w
                foreground_weights[selected_bg_indices[:, 0], 
                                   selected_bg_indices[:, 1], 
                                   selected_bg_indices[:, 2], 
                                   selected_bg_indices[:, 3]] = 1.0
                                   
        else:
            # 特殊情况: 这一帧里完全没有前景
            # 策略: 为了防止 Loss 变成 0 或报错，通常保留所有背景，或者随机采一部分
            # 这里建议保留所有有效背景
            if num_bg > 0:
                foreground_weights[valid_bg_mask] = 1.0

        # 4. 应用权重到 Loss
        # 注意：foreground_weights[mask] 会把它展平成和 loss_conf 一样的 1D 形状
        loss_conf = loss_conf * foreground_weights[mask]
        loss_reg = loss_reg * foreground_weights[mask]


    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.sum() / foreground_weights.sum()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.sum() / foreground_weights.sum()
    else:
        loss_reg = (0.0 * pred).mean()
    
    return loss_conf, loss_grad, loss_reg

def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total

# normal wrapper
def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()

def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids

# grad wrapper
def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


# others
def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim) -> torch.Tensor:
    """在 dim 上做带 mask 的均值，mask 为 0 的位置不计入；若全无有效像素，则返回 0。"""
    mask = mask.to(dtype=x.dtype)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1.0)
    return num / den

def _masked_median_hw(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    在 H,W 维度上做“带 mask 的中位数”。
    x: (B,S,H,W,C)
    mask: (B,S,H,W)  —— 对 C 共享同一 mask
    返回: (B,S,C)
    """
    B, S, H, W, C = x.shape
    N = H * W

    x_flat  = x.reshape(B, S, N, C)                      # (B,S,N,C)
    m_flat  = mask.reshape(B, S, N).unsqueeze(-1) > 0    # (B,S,N,1) -> bool
    valid_n = m_flat.sum(dim=2)                          # (B,S,1)

    # 将无效像素置为 +inf，排序后丢到末尾
    x_inf = torch.where(m_flat, x_flat, torch.full_like(x_flat, float('inf')))
    x_sorted, _ = torch.sort(x_inf, dim=2)               # (B,S,N,C)

    # 中位数下标 k = (n-1)//2；当 n==0 时用 0（随后会得到 +inf，再替换为 0）
    k = (valid_n - 1).clamp_min(0)                       # (B,S,1)
    k = k.unsqueeze(-1).expand(-1, -1, 1, C)             # (B,S,1,C)

    med = x_sorted.gather(dim=2, index=k).squeeze(2)     # (B,S,C)
    # 若该位置没有有效像素，med 会是 +inf，替换为 0
    med = torch.where(torch.isfinite(med), med, torch.zeros_like(med))
    return med

def affine_invariant_normalize(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    对 (H,W) 维度做归一化： (x - median) / mean(|x - median|)
    x:    (B,S,H,W,C)
    mask: (B,S,H,W)
    返回: 归一化结果 x_hat, 中位数 t, 尺度 s   （t、s 形状均为 (B,S,C)）
    """
    t = _masked_median_hw(x, mask)                       # (B,S,C)
    # broadcast 到 (B,S,H,W,C)
    t_b = t.unsqueeze(2).unsqueeze(3)                    # (B,S,1,1,C)

    mad = (x - t_b).abs()                                # (B,S,H,W,C)
    # 对 (H,W) 做带 mask 的平均（对 C 共享同一 mask）
    m = mask.unsqueeze(-1).to(dtype=x.dtype)             # (B,S,H,W,1)
    s = _masked_mean(mad, m, dim=(2,3))                  # (B,S,C)
    s = s.clamp_min(eps)                                 # 防 0

    x_hat = (x - t_b) / s.unsqueeze(2).unsqueeze(3)      # (B,S,H,W,C)
    return x_hat, t, s

def pose_to_ray_map(c2w, H, W, intrinsics=None, intrinsics_image_size=None):
    """
    Args:
        c2w: [B, T, 4, 4]
        H, W: 目标 Ray Map 的高度和宽度 (e.g., 216, 288)
        intrinsics: [B, T, 3, 3] 像素坐标系的内参
        intrinsics_image_size: (H_orig, W_orig) 可选。
                               如果 intrinsics 是对应原图(如1080p)的，而 H,W 是缩放后的(如216p)，
                               这里必须传入原图尺寸，函数会自动缩放内参。
                               如果不传，默认认为 intrinsics 已经匹配了 H, W。
    """
    B, T, _, _ = c2w.shape
    device = c2w.device
    N = B * T
    c2w_flat = c2w.reshape(N, 4, 4) # .reshape比.view更通用，防内存不连续
    
    # 1. 解析并缩放内参
    if intrinsics is not None:
        intrinsics_flat = intrinsics.reshape(N, -1)
        
        # 提取参数
        if intrinsics_flat.shape[-1] == 9: # 3x3 matrix
            K = intrinsics_flat.reshape(N, 3, 3)
            fx, fy = K[:, 0, 0], K[:, 1, 1]
            cx, cy = K[:, 0, 2], K[:, 1, 2]
        else: # assuming [fx, fy, cx, cy]
            fx, fy, cx, cy = intrinsics_flat.unbind(-1)
            
        # [关键步骤] 如果内参对应分辨率 != 目标分辨率，进行缩放
        if intrinsics_image_size is not None:
            H_orig, W_orig = intrinsics_image_size
            scale_x = W / W_orig
            scale_y = H / H_orig
            
            fx = fx * scale_x
            cx = cx * scale_x
            fy = fy * scale_y
            cy = cy * scale_y
            
        # 调整形状以便广播: (N, 1, 1)
        fx, fy, cx, cy = [x.view(N, 1, 1) for x in (fx, fy, cx, cy)]
        
    else:
        # 默认内参处理 (略，同上个回答)
        fx = fy = torch.tensor(W / 2.0, device=device).view(1, 1, 1)
        cx = torch.tensor(W / 2.0, device=device).view(1, 1, 1)
        cy = torch.tensor(H / 2.0, device=device).view(1, 1, 1)

    # 2. 生成网格 (Meshgrid)
    # i: y轴 (H), j: x轴 (W)
    i, j = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), 
                          indexing='ij')
    # i, j 变成 (1, H, W)
    i, j = i.unsqueeze(0), j.unsqueeze(0) 

    # 3. 反投影 (Unprojection)
    # OpenCV 坐标系: x right, y down, z forward
    # 你的内参是像素坐标，所以不用归一化，直接减去 cx, cy 即可
    dirs_x = (j - cx) / fx
    dirs_y = (i - cy) / fy
    dirs_z = torch.ones_like(dirs_x) # 深度为 1 的平面

    # 堆叠: (N, H, W, 3)
    dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)

    # 4. 旋转到世界坐标 (Rotate to World)
    # R: (N, 3, 3)
    R = c2w_flat[:, :3, :3]
    
    # 矩阵乘法: dirs_world = dirs_cam @ R^T
    # (N, HW, 3) @ (N, 3, 3)
    dirs_cam_flat = dirs_cam.reshape(N, -1, 3)
    rays_d = torch.bmm(dirs_cam_flat, R.transpose(1, 2))
    rays_d = rays_d.reshape(N, H, W, 3)

    # 5. 设置原点 (Origin)
    # t: (N, 3) -> (N, H, W, 3)
    t = c2w_flat[:, :3, 3].view(N, 1, 1, 3).expand(-1, H, W, -1)

    # 6. 组合并恢复 B, T
    ray_map = torch.cat([rays_d, t], dim=-1)
    ray_map = ray_map.view(B, T, H, W, 6)
    
    return ray_map