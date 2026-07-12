import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .loss_container import BaseLoss
from .loss_utils import compute_camera_loss, compute_depth_loss, compute_point_loss, compute_trajectory_loss, compute_ray_loss, compute_gsdepth_loss


class CameraPoseLoss(BaseLoss):
    """相机姿态损失"""
    
    def __init__(self, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV", 
                 weight_T=1.0, weight_R=1.0, weight_fl=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoding_type = pose_encoding_type
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
    
    def compute_loss(self, predictions, batch):
        loss_dict = compute_camera_loss(
            predictions, 
            batch, 
            loss_type=self.loss_type,
            gamma=self.gamma,
            pose_encoding_type=self.pose_encoding_type,
            weight_trans=self.weight_T,
            weight_rot=self.weight_R,
            weight_focal=self.weight_fl
        )

        return loss_dict["loss_camera"], loss_dict
    
    @property
    def name(self):
        return f"CameraPoseLoss"


class Point3DLoss(BaseLoss):
    """3D点云损失"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
    
    def compute_loss(self, predictions, batch):
        loss_dict = compute_point_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range,            
        )
        loss = loss_dict["loss_conf_point"] + loss_dict["loss_reg_point"] + loss_dict["loss_grad_point"]
        return loss, loss_dict
    
    @property
    def name(self):
        name = f"Point3DLoss"
        return name

class Trajectory3DLoss(BaseLoss):
    """3D点云轨迹损失"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, foreground_weight=10.0, foreground_prob=-1, use_conf_loss=True, norm="l2"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
        self.foreground_weight = foreground_weight
        self.foreground_prob = foreground_prob
        self.use_conf_loss = use_conf_loss
        self.norm = norm
    
    def compute_loss(self, predictions, batch):
        loss_dict = compute_trajectory_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range,
            foreground_weight=self.foreground_weight,
            use_conf_loss=self.use_conf_loss,
            norm=self.norm,
            foreground_prob=self.foreground_prob if batch["scene_meta"]["dataset_name"][0] == "ue" else -1.0,            
        )
        loss = loss_dict["loss_conf_trajectory"] + loss_dict["loss_reg_trajectory"] + loss_dict["loss_grad_trajectory"]
        # if self.use_velocity_loss:
        #     loss = loss + loss_dict["loss_v_reg_trajectory"] + loss_dict["loss_v_conf_trajectory"]
        return loss, loss_dict
    
    @property
    def name(self):
        name = f"Trajectory3DLoss"
        return name
    
class DepthLoss(BaseLoss):
    """深度损失"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=0.98, norm="l2"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
        self.norm = norm

    def compute_loss(self, predictions, batch):
        loss_dict = compute_depth_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range,
            norm=self.norm
        )
        loss = loss_dict["loss_conf_depth"] + loss_dict["loss_reg_depth"] + loss_dict["loss_grad_depth"]

        return loss, loss_dict
    
    @property
    def name(self):
        name = f"DepthLoss"
        return name

    
class RayLoss(BaseLoss):
    """深度损失"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=0.98, use_conf_loss=False, norm="l1"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
        self.use_conf_loss = use_conf_loss
        self.norm = norm

    def compute_loss(self, predictions, batch):
        loss_dict = compute_ray_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range,
            use_conf_loss=self.use_conf_loss,
            norm=self.norm,
        )
        loss = loss_dict["loss_conf_ray"] + loss_dict["loss_reg_ray"] + loss_dict["loss_grad_ray"]

        return loss, loss_dict
    
    @property
    def name(self):
        name = f"RayLoss"
        return name

class GSDepthLoss(BaseLoss):
    """GS深度损失"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=0.98):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range

    def compute_loss(self, predictions, batch):
        loss_dict = compute_gsdepth_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range
        )
        loss = loss_dict["loss_conf_gsdepth"] + loss_dict["loss_reg_gsdepth"] + loss_dict["loss_grad_gsdepth"]

        return loss, loss_dict
    
    @property
    def name(self):
        name = f"GSDepthLoss"
        return name


# class DynamicScoreLoss(BaseLoss):
#     """dynamic score损失"""

#     def __init__(self, movement_threshold=0.01, only_pts3d=False, only_token=False, pts3d_mask=False, token_mask=False,
#         loss_weight=[1.0, 1.0]):
#         super().__init__()
#         self.movement_threshold = movement_threshold
#         self.only_pts3d = only_pts3d
#         self.only_token = only_token
#         self.pts3d_mask = pts3d_mask
#         self.token_mask = token_mask
#         self.loss_weight = loss_weight

#     def compute_loss(self, predictions, batch):
#         # TODO: sometimes we should use foreground_mask
#         # eg omnigame/dynamic replica dataset
#         # TODO: support loss to let score become 0-1
#         if not self.pts3d_mask:
#             pts3d_dynamic_score = predictions["pts3d_dynamic_score"]  # [b, im, h, w]
#         else:
#             pts3d_dynamic_score = predictions["pts3d_dynamic_mask"]  # [b, im, h, w]
#         trajectory = batch["trajectory"]  # [b, im, t, h, w, c]

#         # if batch.get("use_foreground_mask_as_dynamic_mask", None) is not None:
#         # TODO: use dynamic_mask if possible
#         # 1. 计算帧间运动欧氏距离
#         # shift trajectory by 1 along time and compute distance
#         displacement = torch.norm(trajectory[:, :, 1:] - trajectory[:, :, :-1], dim=-1)  # [b, im, t-1, h, w]

#         # 2. 求 t 维度最大位移（防止短暂抖动被误判）
#         max_motion = displacement.max(dim=2).values  # [b, im, h, w]
        
#         # 3. 生成动态标签mask
#         pts3d_target_mask = (max_motion > self.movement_threshold).float()  # 1=动态, 0=静止

#         # # 4. BCE loss（动态概率映射mask）
#         # with torch.amp.autocast('cuda', dtype=torch.float32, enabled=False):
#         #     loss = F.binary_cross_entropy(dynamic_score.float(), target_mask.float())
#         loss_pts3d = F.mse_loss(pts3d_dynamic_score.float(), pts3d_target_mask.float())

#         # calcualte token-level dynamic mask
#         if not self.token_mask:
#             token_dynamic_score = predictions["dynamic_token_score"] # [b im l]
#         else:
#             token_dynamic_score = predictions["dynamic_token_mask"] # [b im l]
#         patch_size = ((pts3d_target_mask.shape[-2] * pts3d_target_mask.shape[-1]) // token_dynamic_score.shape[-1])**0.5
#         patch_size = int(patch_size)
#         resized_size = (int(pts3d_target_mask.shape[-2] // patch_size), int(pts3d_target_mask.shape[-1] // patch_size))
        
#         # token_target_mask = F.interpolate(pts3d_target_mask, size=resized_size, mode='nearest')
#         token_target_mask = F.max_pool2d(
#             pts3d_target_mask.float(),
#             kernel_size=patch_size,
#             stride=patch_size
#         )  # [B, im, H//p, W//p]

#         token_target_mask = token_target_mask.reshape(token_target_mask.shape[0], token_target_mask.shape[1], -1)
#         loss_token = F.mse_loss(token_dynamic_score.float(), token_target_mask.float())

#         if self.only_pts3d:
#             loss = loss_pts3d
#         elif self.only_token:
#             loss = loss_token
#         else:
#             loss = self.loss_weight[0] * loss_pts3d + self.loss_weight[1] * loss_token
#         # TODO: tmp_debug
#         # loss = loss_token

#         loss_dict = {
#             "loss_dynamic_score": loss.item(),
#             "loss_pts3d_dynamic_score": loss_pts3d.item(),
#             "loss_token_dynamic_score": loss_token.item(),
#             "dynamic_ratio": pts3d_target_mask.mean().item(),  # 数据里动态的比例
#         }
#         return loss, loss_dict
    
#     @property
#     def name(self):
#         return "DynamicScoreLoss"



class DynamicScoreLoss(BaseLoss):
    """dynamic score损失 (Handle NaN and Foreground Fallback)"""

    def __init__(self, movement_threshold=0.01, pts3d_mask=False, token_mask=False, loss_weight=[1.0, 1.0], use_bce=False):
        super().__init__()
        self.movement_threshold = movement_threshold
        self.pts3d_mask = pts3d_mask
        self.token_mask = token_mask
        self.loss_weight = loss_weight
        self.use_bce = use_bce
    
    def _build_dummy_loss(self, pts3d_dynamic_pred, token_dynamic_pred):
        dummy_loss = (
            (0.0 * pts3d_dynamic_pred).mean()
            + (0.0 * token_dynamic_pred).mean()
        )

        loss_dict = {
            "loss_dynamic_score": dummy_loss.item(),
            "loss_pts3d_dynamic_score": dummy_loss.item(),
            "loss_token_dynamic_score": dummy_loss.item(),
            "dynamic_ratio": 0.0,
            "valid_supervision_ratio": 0.0,
            "annotated_supervision_ratio": 0.0,
        }

        return dummy_loss, loss_dict

    def compute_loss(self, predictions, batch):
        # -----------------------------------------------------------------
        # 1. 准备 Predictions
        # -----------------------------------------------------------------

        if not self.pts3d_mask:
            pts3d_dynamic_pred = predictions["pts3d_dynamic_score"]  # [b, im, h, w]
        else:
            pts3d_dynamic_pred = predictions["pts3d_dynamic_mask"]
        
        if not self.token_mask:
            token_dynamic_pred = predictions["dynamic_token_score"]  # [b, im, l]
        else:
            token_dynamic_pred = predictions["dynamic_token_mask"]

        # -----------------------------------------------------------------
        # 2. 处理 Trajectory GT (Handle NaN)
        # -----------------------------------------------------------------
        # import ipdb
        # ipdb.set_trace()
        if "trajectory" not in batch:
            return self._build_dummy_loss(pts3d_dynamic_pred, token_dynamic_pred)

        trajectory = batch["trajectory"]  # [b, im, t, h, w, 3]
        
        # 计算帧间差分: [b, im, t-1, h, w, 3]
        diff = trajectory[:, :, 1:] - trajectory[:, :, :-1]
        
        # 计算位移模长 (若有NaN，结果仍为NaN) -> [b, im, t-1, h, w]
        displacement = torch.norm(diff, dim=-1)
        
        # 标记哪些时间步的位移是有效的 (非NaN)
        valid_steps_mask = ~torch.isnan(displacement) # [b, im, t-1, h, w]
        
        # 填充NaN为-1 (为了方便求max，不影响结果，因为我们会用mask过滤)
        displacement_filled = torch.nan_to_num(displacement, nan=-1.0)
        
        # 求时间维度上的最大位移
        max_motion = displacement_filled.amax(dim=2)  # [b, im, h, w]
        
        # 判断该像素点是否有足够的数据来计算动态 (即至少有一个有效的时间间隔)
        has_valid_traj_data = valid_steps_mask.any(dim=2) # [b, im, h, w] bool

        # -----------------------------------------------------------------
        # 3. 生成 Target 和 Loss Mask (Fallback Logic)
        # -----------------------------------------------------------------
        # 初始化 Target 和 Loss Mask
        # Target: 1=动态, 0=静态
        target_mask = torch.zeros_like(max_motion).float()
        # Loss Mask: 1=计算Loss, 0=忽略该像素
        loss_validity_mask = torch.zeros_like(max_motion).float()

        # === 逻辑 A: 轨迹数据有效 ===
        # 如果轨迹有效，根据阈值判断动态
        is_dynamic_traj = (max_motion > self.movement_threshold).float()
        
        # 填入 valid 部分的数据
        target_mask[has_valid_traj_data] = is_dynamic_traj[has_valid_traj_data]
        loss_validity_mask[has_valid_traj_data] = 1.0

        # === 逻辑 B: 轨迹数据无效 (全NaN 或 无法插值) ===
        # 找到无效的区域
        invalid_area = ~has_valid_traj_data
    
        if invalid_area.any():
            scene_name = batch["scene_meta"]["dataset_name"][0]
            # omnigame 的 foreground_mask 不太准
            if "foreground" in batch and scene_name not in ["omnigame"]:
                fg_mask = batch["foreground"].float() # [b, im, h, w]
                # 如果有前景mask，用前景mask作为GT (前景=动态)
                target_mask[invalid_area] = fg_mask[invalid_area]
                # 这些区域也需要计算loss
                loss_validity_mask[invalid_area] = 1.0
            else:
                # 如果没有前景mask，且轨迹也是NaN，则无法判定，Loss Mask保持为0 (忽略)
                pass

        # -----------------------------------------------------------------
        # 4. 计算 Pixel-level Loss (PTS3D)
        # -----------------------------------------------------------------
        # 使用 reduction='none' 保留维度，然后应用 mask
        if self.use_bce:
            with torch.cuda.amp.autocast(enabled=False):
                loss_pixel_raw = F.binary_cross_entropy(
                    pts3d_dynamic_pred.float().clamp(1e-6, 1-1e-6), # 加上 clamp 防止 NaN
                    target_mask.float(), 
                    reduction='none'
            )
        else:
            loss_pixel_raw = F.mse_loss(pts3d_dynamic_pred.float(), target_mask.float(), reduction='none')
        # Apply mask: 只计算有效区域的 loss
        # 为了避免除以0，分母加个 eps
        loss_pts3d = (loss_pixel_raw * loss_validity_mask).sum() / (loss_validity_mask.sum() + 1e-6)

        # -----------------------------------------------------------------
        # 5. 计算 Token-level Loss
        # -----------------------------------------------------------------
        # 计算 patch size
        patch_size = ((target_mask.shape[-2] * target_mask.shape[-1]) // token_dynamic_pred.shape[-1])**0.5
        patch_size = int(patch_size)
        
        # Downsample Target Mask: Max Pooling
        # 语义：Patch内只要有一个像素是动态，Token即视为动态
        token_target = F.max_pool2d(
            target_mask.float(),
            kernel_size=patch_size,
            stride=patch_size
        ) # [b, im, h_p, w_p]
        
        # Downsample Loss Validity Mask: Max Pooling (或者 Avg)
        # 语义：Patch内只要有一个像素是有效的GT，我们就计算这个Token的Loss
        token_validity_mask = F.max_pool2d(
            loss_validity_mask.float(),
            kernel_size=patch_size,
            stride=patch_size
        ) # [b, im, h_p, w_p]
        
        # Flatten
        token_target = token_target.flatten(2) # [b, im, l]
        token_validity_mask = token_validity_mask.flatten(2) # [b, im, l]

        # Token Loss
        if self.use_bce:
            with torch.cuda.amp.autocast(enabled=False):
                loss_token_raw = F.binary_cross_entropy(
                    token_dynamic_pred.float(), 
                    token_target.float(), 
                    reduction='none'
                )
        else:
            loss_token_raw = F.mse_loss(token_dynamic_pred.float(), token_target.float(), reduction='none')
            
        loss_token = (loss_token_raw * token_validity_mask).sum() / (token_validity_mask.sum() + 1e-6)

        # -----------------------------------------------------------------
        # 6. 汇总
        # -----------------------------------------------------------------
        loss = self.loss_weight[0] * loss_pts3d + self.loss_weight[1] * loss_token

        loss_dict = {
            "loss_dynamic_score": loss.item(),
            "loss_pts3d_dynamic_score": loss_pts3d.item(),
            "loss_token_dynamic_score": loss_token.item(),
            # 统计动态比例时，只统计有效区域
            "dynamic_ratio": (target_mask * loss_validity_mask).sum().item() / (loss_validity_mask.sum().item() + 1e-6),
            "valid_supervision_ratio": loss_validity_mask.mean().item() # 监控有多少比例的像素参与了Loss计算
        }
        
        return loss, loss_dict

class Rot6DRegLoss(BaseLoss):
    """6D旋转表示的正则化Loss，用于约束两个3D向量接近单位正交"""
    
    def __init__(self,):
        """
        :param weight: 正则项的权重系数（乘到总loss上）
        """
        super().__init__()

    def compute_loss(self, predictions, batch):
        """
        predictions: dict，必须包含 "rot6d" -> [*, 6] 张量
        batch: 这里没有用到，但为了和BaseLoss接口一致保留
        """
        if "motion_pred" not in predictions:
            raise KeyError("Predictions dict 缺少 'motion_pred' 张量")
            
        motion_pred = predictions["motion_pred"]  # [..., 6]
        x1 = motion_pred[..., 0:3]
        y1 = motion_pred[..., 3:6]

        # 长度正则（接近1）
        norm_loss = ((torch.norm(x1, dim=-1) - 1)**2
                    + (torch.norm(y1, dim=-1) - 1)**2).mean()

        # 正交性正则（点积平方）
        ortho_loss = (torch.sum(F.normalize(x1, dim=-1) *
                                F.normalize(y1, dim=-1),
                                dim=-1)**2).mean()

        # 总loss
        loss_val = norm_loss + ortho_loss

        loss_dict = {
            "loss_rot6d_reg": loss_val.item(),
            "rot6d_norm_loss": norm_loss.item(),
            "rot6d_ortho_loss": ortho_loss.item(),
        }

        return loss_val, loss_dict

    @property
    def name(self):
        return "Rot6DRegLoss"