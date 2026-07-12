import torch
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
import torch.nn as nn
        
import math
from einops import rearrange, repeat

from torch.utils.checkpoint import checkpoint

import sys

# Add the build library path
sys.path.insert(0, '/data/tmp/dependencies/Deformable_DETR')

from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
from src.models.components.custom_attention import fixed_offset_similarity_cuda_op, fixed_offset_attention_op

# motion vector weighting, use fixed offset attention / ms_deform
# work on this
class GaussianTrajectoryMod(torch.nn.Module):
    """Trajectory Generator"""

    def __init__(
        self,
        num_point=9,
        patch_size=14,
        motion_dim=9,
        # sampling
        fixed_offset_level=[0],
        ms_deform_level=[1],
        interpolate_dynamic_mask=False,
        resize_factor=2,
        # training
        use_time_info=True,
        detach_pts3d=False,
        use_checkpointing=False,
    ):
        super(TrajectoryMod, self).__init__()

        self.num_point = num_point
        self.patch_size = patch_size
        self.motion_dim = motion_dim

        self.fixed_offset_level = fixed_offset_level
        self.ms_deform_level = ms_deform_level

        self.interpolate_dynamic_mask = interpolate_dynamic_mask
        self.use_checkpointing = use_checkpointing
        self.resize_factor = resize_factor

        self.detach_pts3d = detach_pts3d
        self.use_time_info = use_time_info

        if self.motion_dim != 9 and self.motion_dim != 3:
            self.motion_proj = nn.Linear(motion_dim, 9)
            torch.nn.init.zeros_(self.motion_proj.weight)

    def forward(self, pts3d, offset_output, motion_field_list=None, dynamic_token_info=None, time_info=None):
        """
        Args:
            motion_field_list: list, [b im t (hh ww) c]
            offset_output: dict 
                sampling_offsets: [optional] [b im h w num_level*num_points*2] 
                sampling_weights: [b im h w num_level*num_points]
                sampling_weights_logits: logits

            pts3d: [b, im, h, w, 3]
            time_info: [b im], representing global_time_index of pts3d
        Returns:
            pts3d_traj: [b, im, t, h, w, 3]
        """

        b, im, h, w, _ = pts3d.shape
        t = motion_field_list[0].shape[2]
        hp = h // self.patch_size
        wp = w // self.patch_size
        resize_factor = self.resize_factor
        
        num_level = len(motion_field_list)

        if self.detach_pts3d:
            pts3d = pts3d.clone().detach()
            
        sampling_weights = offset_output["sampling_weights"]
        sampling_weights = rearrange(sampling_weights, "b im h w (nl np) -> b im h w nl np", h=h, w=w, nl=num_level, np=self.num_point)
        motion_field_list = [rearrange(motion_field, (" b im t (hp wp) c -> b im hp wp (t c)"), hp=hp*resize_factor, wp=wp*resize_factor) for motion_field in motion_field_list]

        # concat dynamic_score / dynamic_mask
        if self.interpolate_dynamic_mask:
            dynamic_input = dynamic_token_info["dynamic_mask"]
        else:
            dynamic_input = dynamic_token_info["dynamic_score"]
        dynamic_input = rearrange(dynamic_input, "b im (hp wp) -> b im hp wp", hp=hp*resize_factor, wp=wp*resize_factor)

        motion_field_list = [torch.cat([motion_field, dynamic_input.unsqueeze(-1)], dim=-1) for motion_field in motion_field_list]

        fixed_offset_weighted_values = None
        if len(self.fixed_offset_level) > 0:
            fixed_offset_motion_field_list = []
            fixed_offset_sampling_weights = []
            
            for level_idx in self.fixed_offset_level:
                fixed_offset_motion_field_list.append(motion_field_list[level_idx])
                fixed_offset_sampling_weights.append(sampling_weights[:, :, :, :, level_idx])
            fixed_offset_sampling_weights = torch.stack(fixed_offset_sampling_weights, dim=4)
            
            if self.use_checkpointing:
                fixed_offset_weighted_values = checkpoint(
                    lambda *args: self.fixed_offset_attention_weighted_cuda(
                        list(args[:-1]),  # 前面的都是 motion_field_list 的元素
                        args[-1]          # 最后一个是 sampling_weights
                    ),
                    *fixed_offset_motion_field_list,
                    fixed_offset_sampling_weights,
                )
            else:
                fixed_offset_weighted_values = self.fixed_offset_attention_weighted_cuda(fixed_offset_motion_field_list, fixed_offset_sampling_weights)
        
        ms_deform_weighted_values = None
        if len(self.ms_deform_level) > 0:
            assert "sampling_offsets" in offset_output, "msdeform_attention requires sampling_offsets"
            sampling_offsets = offset_output["sampling_offsets"]
            sampling_offsets = rearrange(sampling_offsets, "b im h w (nl np d) -> b im h w nl np d", nl=len(self.ms_deform_level), np=self.num_point, d=2)
            
            ms_deform_motion_field_list = []
            ms_deform_sampling_offsets, ms_deform_sampling_weights = [], []
        
            for level_idx in self.ms_deform_level:
                ms_deform_motion_field_list.append(motion_field_list[level_idx])
                ms_deform_sampling_weights.append(sampling_weights[:, :, :, :, level_idx])
            ms_deform_sampling_weights = torch.stack(ms_deform_sampling_weights, dim=4)
            
            # NOTE: Pass `num_time=t` to the function
            if self.use_checkpointing:
                ms_deform_weighted_values = checkpoint(
                    lambda *args: self.msdeform_attention_weighted_cuda(
                        list(args[:-2]),  # value_list
                        args[-2],         # sampling_offsets
                        args[-1],         # sampling_weights
                        t                 # num_time
                    ),
                    *ms_deform_motion_field_list,
                    sampling_offsets,
                    ms_deform_sampling_weights,
                )
            else:
                ms_deform_weighted_values = self.msdeform_attention_weighted_cuda(
                    ms_deform_motion_field_list, 
                    sampling_offsets, 
                    ms_deform_sampling_weights,
                    num_time=t
                )

        if fixed_offset_weighted_values is not None and ms_deform_weighted_values is not None:
            weighted_values = fixed_offset_weighted_values + ms_deform_weighted_values
        elif fixed_offset_weighted_values is not None:
            weighted_values = fixed_offset_weighted_values
        elif ms_deform_weighted_values is not None:
            weighted_values = ms_deform_weighted_values
        else:
            raise ValueError("fixed_offset_weighted_values and ms_deform_weighted_values cannot be both None")


        weighted_motions = weighted_values[..., :-1]
        weighted_dynamic_score = weighted_values[..., -1]

        if self.motion_dim != 9 and self.motion_dim != 3:
            weighted_motions = rearrange(weighted_motions, "b im h w (t d) -> b im t h w d", t=t)
            pts3d_transform = self.motion_proj(weighted_motions)
        else:
            pts3d_transform = rearrange(weighted_motions, "b im h w (t d) -> b im t h w d", t=t)
        pts3d_dynamic_score = weighted_dynamic_score
      
        # TODO: only dynamic point need transform?  
        pts3d_trajectory = self.transform_pts3d(pts3d, pts3d_transform, time_info=time_info if self.use_time_info else None)

        pts3d_static_traj = pts3d.unsqueeze(2).expand_as(pts3d_trajectory)
        w_dynamic = pts3d_dynamic_score.unsqueeze(2).unsqueeze(-1)
        pts3d_trajectory = (
            w_dynamic * pts3d_trajectory +
            (1 - w_dynamic) * pts3d_static_traj
        )
            
        output = {}
        output["pts3d_trajectory"] = pts3d_trajectory
        output["pts3d_dynamic_score"] = pts3d_dynamic_score
        
        return output

    def transform_pts3d(self, pts3d, motion_pred, time_info=None):
        """pts3d_traj generator"""
        b, im, t, h, w, _ = motion_pred.shape
        
        # Split rotation and translation
        if motion_pred.shape[-1] == 9:
            cont_6d = motion_pred[..., :6]  # [b, im, t, h, w, 6]
            trans = motion_pred[..., 6:9]   # [b, im, t, h, w, 3]
        else:
            cont_6d = None
            trans = motion_pred

        # Convert 6d to rotation matrix
        if cont_6d is not None:
            rmat = self.cont_6d_to_rmat(cont_6d)  # [b, im, t, h, w, 3, 3]
        
        # Normalize by reference time
        if time_info is not None:
            # Extract reference frame at time_info
            batch_idx = torch.arange(b, device=motion_pred.device)[:, None]
            im_idx = torch.arange(im, device=motion_pred.device)[None, :]
            
            if cont_6d is not None:
                rmat_ref = rmat[batch_idx, im_idx, time_info]  # [b, im, h, w, 3, 3]
            trans_ref = trans[batch_idx, im_idx, time_info]  # [b, im, h, w, 3]
            
            # Normalize: R_norm = R @ R_ref^T, T_norm = T - T_ref
            if cont_6d is not None:
                rmat_ref_inv = rmat_ref.transpose(-2, -1).unsqueeze(2)  # [b, im, 1, h, w, 3, 3]
                rmat = torch.matmul(rmat, rmat_ref_inv)
            trans = trans - trans_ref.unsqueeze(2)
        
        # Expand pts3d along time dimension
        pts3d_expanded = pts3d.unsqueeze(2).expand(b, im, t, h, w, 3)
        
        # Apply rotation and translation
        if cont_6d is not None:
            pts3d_rotated = torch.einsum('...ij,...j->...i', rmat, pts3d_expanded)
            pts3d_traj = pts3d_rotated + trans
        else:
            pts3d_traj = pts3d_expanded + trans
        
        return pts3d_traj
        
    def cont_6d_to_rmat(self, cont_6d, eps=1e-3):
        """:param cont_6d: 6d vector (*, 6) -> rotation matrix (*, 3, 3)"""
        x1 = cont_6d[..., 0:3]
        y1 = cont_6d[..., 3:6]
        x = x1 / torch.clamp(torch.norm(x1, dim=-1, keepdim=True), min=eps)
        proj = (y1 * x).sum(dim=-1, keepdim=True) * x
        u2 = y1 - proj
        y = u2 / torch.clamp(torch.norm(u2, dim=-1, keepdim=True), min=eps)
        z = torch.cross(x, y, dim=-1)
        return torch.stack([x, y, z], dim=-1)  # [*, 3, 3]
    
    def fixed_offset_attention_weighted_cuda(self, value_list, attn_weight):
        """
        value_list: list of [B, im, H_q, W_q, C]，最后一个通道为 dynamic_score
        attn_weight: [B, im, H_hr, W_hr, num_level, num_point_total]
        返回: [B, im, H_hr, W_hr, C]
        """
        b, im = attn_weight.shape[:2]
        # hh, ww = value_list[0].shape[2:4] # unused
        value_stacked = torch.stack(value_list, dim=1)
        value_reshaped = rearrange(value_stacked, "b l im h w c -> (b im) l c h w")
        attn_reshaped = rearrange(attn_weight, "b im h w l np -> (b im) h w l np")
        
        # fixed_offset_attention_op defined elsewhere (custom cuda kernel)
        output = fixed_offset_attention_op(value_reshaped, attn_reshaped) 
        output = rearrange(output, "(b im) c h w -> b im h w c", b=b, im=im)
        return output
    
    def msdeform_attention_weighted_cuda(self, value_list, attn_offset, attn_weight, num_time):
        """
        value_list: list of [B, im, H_q, W_q, C_total]
        C_total = t * motion_dim + 1
        
        attn_offset: [B, im, H_hr, W_hr, num_level, num_point, 2]
        attn_weight: [B, im, H_hr, W_hr, num_level, num_point]
        
        num_time: 这里的 num_time 参数不再用于 head 维度，仅作为逻辑参考
        """
        b, im, h, w, num_level, _ = attn_weight.shape
        hp, wp = value_list[0].shape[2:4]
        device = value_list[0].device
        c_total = value_list[0].shape[-1]
        
        # ==========================================
        # 1. 准备采样点 (保持 num_head=1，极省显存)
        # ==========================================
        
        # [B*im, H*W, 1, L, P, 2]
        attn_offset = attn_offset.view(b*im, h*w, 1, num_level, self.num_point, 2) 
        attn_weight = attn_weight.view(b*im, h*w, 1, num_level, self.num_point)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        reference_points = torch.stack([grid_x, grid_y], dim=-1).view(1, h * w, 1, 2)
        reference_points = reference_points[:, :, None, :, None, :] 

        input_spatial_shapes = torch.tensor([[hp, wp]] * num_level, dtype=torch.float32, device=device)
        offset_normalizer = torch.stack([input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]], dim=-1)
        offset_normalizer = offset_normalizer[None, None, None, :, None, :]

        # 这里的 sampling_locations 只有 1 个 head，不需要 expand，不需要 contiguous 复制
        sampling_locations = reference_points + attn_offset / offset_normalizer
        sampling_locations = sampling_locations.clamp(0, 1)

        input_spatial_shapes_int = torch.as_tensor([[hp, wp]] * num_level, dtype=torch.long, device=device)
        level_start_index = torch.cat((
            input_spatial_shapes_int.new_zeros((1,)),
            input_spatial_shapes_int.prod(1).cumsum(0)[:-1]
        ))
        im2col_step = 2

        # ==========================================
        # 2. 准备 Value (Flatten Level, 保持 C_total)
        # ==========================================
        # value: [b im hp wp C] -> [(b im) (sum_hp_wp) C]
        value_list_flat = [rearrange(v, "b im h w c -> (b im) (h w) c") for v in value_list]
        value_flat = torch.cat(value_list_flat, dim=1) 
        
        # ==========================================
        # 3. Channel Chunking (分批处理)
        # ==========================================
        # MSDeformAttn 的 channel 上限通常是 1024，为了安全我们设为 512 或 256
        MAX_CHANNEL_CHUNK = 512
        
        # 预分配输出显存 [b, im, h, w, C_total]
        # 注意：这里我们按照原始形状预分配，避免最后 cat 产生显存峰值
        output = torch.empty(b, im, h, w, c_total, dtype=value_list[0].dtype, device=device)
        
        # 循环处理 channel
        for start_idx in range(0, c_total, MAX_CHANNEL_CHUNK):
            end_idx = min(start_idx + MAX_CHANNEL_CHUNK, c_total)
            
            # 切片 (Slicing creates a view, low overhead)
            value_chunk = value_flat[..., start_idx:end_idx].unsqueeze(2).contiguous()
            
            with torch.cuda.amp.autocast(enabled=False):
                # num_head=1, channels=chunk_size
                # 显存消耗极小，只计算当前 chunk
                out_chunk = MSDeformAttnFunction.apply(
                    value_chunk.float(),
                    input_spatial_shapes_int,
                    level_start_index,
                    sampling_locations.float(), # Shared! No copy!
                    attn_weight.float(),        # Shared! No copy!
                    im2col_step
                )
            
            # 转换类型并写回预分配的 Tensor
            # Out_chunk: [(b im) (h w) C_chunk] -> reshaped -> [b im h w C_chunk]
            output[..., start_idx:end_idx] = rearrange(
                out_chunk.to(value_list[0].dtype), 
                "(b im) (h w) c -> b im h w c", b=b, im=im, h=h, w=w
            )

        return output