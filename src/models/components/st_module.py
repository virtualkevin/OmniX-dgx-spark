from typing import Callable, List, Any, Tuple, Dict
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

# from flash_attn import flash_attn_varlen_func

from einops import rearrange, repeat

from src.models.components.embedding import SinusoidPosEmbed
from depth_anything_3.model.dinov2.layers.attention import Attention
from depth_anything_3.model.dinov2.layers.mlp import Mlp
from depth_anything_3.model.dinov2.layers.layer_scale import LayerScale
from depth_anything_3.model.dinov2.layers.drop_path import DropPath
from depth_anything_3.model.dinov2.layers.block import drop_add_residual_stochastic_depth
from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D, PositionGetter

from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous

from flash_attn_interface import flash_attn_func as flash_attn_fuc_v3

from torch.utils.checkpoint import checkpoint


class AttentionMask(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, K, K] → 自动广播到 H

        if attn_mask is None and q.dtype == torch.bfloat16 or q.dtype == torch.float16:
            if q.is_contiguous():
                q = q.transpose(1, 2)
            else:
                q = q.transpose(1, 2).contiguous()
            
            if k.is_contiguous():
                k = k.transpose(1, 2)
            else:
                k = k.transpose(1, 2).contiguous()
            
            if v.is_contiguous():
                v = v.transpose(1, 2)
            else:
                v = v.transpose(1, 2).contiguous()
            
            x = flash_attn_fuc_v3(q, k, v)
            if x.is_contiguous():
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).contiguous()
        else:
            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionMask(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # 修改 1: 拆分 Q 和 KV 的投影层
        # Query 来自 x (decoder / target)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Key 和 Value 来自 memory (encoder / source)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, memory: Tensor, pos=None, pos_memory=None, attn_mask=None) -> Tensor:
        # x: [Batch, N, C] (Query序列)
        # memory: [Batch, M, C] (Key/Value序列，M 可以不等于 N)
        B, N, C = x.shape
        M = memory.shape[1]

        # 修改 2: 分别生成 Q 和 K, V
        
        # 1. 生成 Query
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, N, D]
        
        # 2. 生成 Key, Value (从 memory 中)
        kv = self.kv(memory).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # k, v shape: [B, H, M, D]

        # Norm
        q, k = self.q_norm(q), self.k_norm(k)

        # 修改 3: RoPE 处理 (Query 用 x 的位置，Key 用 memory 的位置)
        if self.rope is not None:
            # 假设 rope 接受 (tensor, positions)
            q = self.rope(q, pos)
            k = self.rope(k, pos_memory) 

        # Attention Mask 处理
        # Cross Attention 的 mask 通常用于屏蔽 padding 的 memory，形状通常是 [B, 1, N, M] 或 [B, 1, 1, M]
        if attn_mask is not None:
            if attn_mask.dim() == 2: # [B, M] -> [B, 1, 1, M]
                attn_mask = attn_mask[:, None, None, :]
            elif attn_mask.dim() == 3: # [B, N, M] -> [B, 1, N, M]
                attn_mask = attn_mask.unsqueeze(1)

        if attn_mask is None and q.dtype == torch.bfloat16 or q.dtype == torch.float16:
            if q.is_contiguous():
                q = q.transpose(1, 2)
            else:
                q = q.transpose(1, 2).contiguous()
            
            if k.is_contiguous():
                k = k.transpose(1, 2)
            else:
                k = k.transpose(1, 2).contiguous()
            
            if v.is_contiguous():
                v = v.transpose(1, 2)
            else:
                v = v.transpose(1, 2).contiguous()
            
            x = flash_attn_fuc_v3(q, k, v)
            if x.is_contiguous():
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).contiguous()
        else:
            if self.fused_attn:
                # F.scaled_dot_product_attention 自动处理维度匹配
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                q = q * self.scale
                # [B, H, N, D] @ [B, H, D, M] -> [B, H, N, M]
                attn = q @ k.transpose(-2, -1)
                
                if attn_mask is not None:
                    attn = attn + attn_mask

                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                # [B, H, N, M] @ [B, H, M, D] -> [B, H, N, D]
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn1_class: Callable[..., nn.Module] = AttentionMask, # 这里传入 AttentionMask 类 (Self-Attn)
        attn2_class: Callable[..., nn.Module] = CrossAttentionMask, # 这里传入 CrossAttentionMask 类
        ffn_layer: Callable[..., nn.Module] = Mlp, # MLP
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        use_temporal_attn: bool = True, # 可选项开关
        use_spatial_attn: bool = True, # 可选开关
        use_cross_attn: bool = True, # 可选开关
        use_ref_temporal_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_temporal_attn = use_temporal_attn
        self.use_spatial_attn = use_spatial_attn
        self.use_ref_temporal_attn = use_ref_temporal_attn
        self.use_cross_attn = use_cross_attn
        
        # 1. Temporal Self-Attention (Optional)
        if self.use_temporal_attn:
            self.norm1_tem = norm_layer(dim)
            self.attn_temporal = attn1_class(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, 
                fused_attn=fused_attn, rope=rope
            )
            self.ls1_tem = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
            self.drop_path1_tem = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.use_spatial_attn:
            self.norm1_sp = norm_layer(dim)
            self.attn_spatial = attn1_class(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, 
                fused_attn=fused_attn, rope=rope
            )
            self.ls1_sp = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
            self.drop_path1_sp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 2. Cross-Attention
        if self.use_cross_attn:
            self.norm2 = norm_layer(dim)
            self.attn_cross = attn2_class(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, 
                fused_attn=fused_attn, rope=rope
            )
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 3. FFN
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, 
        x: Tensor, 
        memory: Tensor, 
        pos: Tensor, 
        memory_pos: Tensor, 
        use_mask_attn: bool = False,
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
        use_cross_attn: bool = True,
    ) -> Tensor:
        """
            x: [b, im, t, k, c]
            pos: [b, im, t, k, 2] (或对应维度的 embedding)
            memory: [b, im, l, c]
            memory_pos: [b, im, l, 2]
            use_mask_attn: bool
            use_temporal_attn: bool
        """
        b, im, t, k, _ = x.shape
        l = memory.shape[2]
  
        # --- 1. Temporal Self-Attention ---
        if use_temporal_attn:
            if not self.use_ref_temporal_attn:
                x_t = rearrange(x, "b im t k c -> (b im k) t c")
                pos_t = rearrange(pos, "b im t k d -> (b im k) t d") if pos is not None else None
                
                # Residual + Norm
                shortcut = x_t
                x_t = self.norm1_tem(x_t)
            
                x_t = self.attn_temporal(x_t, pos=pos_t)
                
                x_t = self.ls1_tem(x_t)
                x_t = shortcut + self.drop_path1_tem(x_t)
                
                x = rearrange(x_t, "(b im k) t c -> b im t k c", b=b, im=im, k=k)
            else:

                x_t = rearrange(x, "b im t k c -> (b im k) t c")
                pos_t = rearrange(pos, "b im t k d -> (b im k) t d") if pos is not None else None

                shortcut = x_t
                x_t = self.norm1_tem(x_t)

                ref_idx = torch.arange(im, device=x.device)
                ref_x = x[:, ref_idx, ref_idx, :, :] 
                ref_pos = pos[:, ref_idx, ref_idx, :, :]

                ref_value = rearrange(ref_x, "b im k c -> (b im k) c")
                ref_value = ref_value.unsqueeze(1)
                ref_value_pos = rearrange(ref_pos, "b im k c -> (b im k) c")
                ref_value_pos = ref_value_pos.unsqueeze(1)


                x_t = self.attn_temporal(x=x_t, memory=ref_value, pos=pos_t, pos_memory=ref_value_pos)
                x_t = self.ls1_tem(x_t)
                x_t = shortcut + self.drop_path1_tem(x_t)

                x = rearrange(x_t, "(b im k) t c -> b im t k c", b=b, im=im, k=k)

        if use_spatial_attn:
            x_sp = rearrange(x, "b im t k c -> (b t) (im k) c")
            pos_sp = rearrange(pos, "b im t k d -> (b t) (im k) d") if pos is not None else None
            # Residual + Norm
            shortcut = x_sp
            x_sp = self.norm1_sp(x_sp)
        
            x_sp = self.attn_spatial(x_sp, pos=pos_sp)
            
            x_sp = self.ls1_sp(x_sp)
            x_sp = shortcut + self.drop_path1_sp(x_sp)
            
            x = rearrange(x_sp, "(b t) (im k) c -> b im t k c", b=b, im=im, k=k)

        # --- 2. Cross-Attention (x attends to memory) ---
        if self.use_cross_attn:
            if use_mask_attn:
                x_cross = rearrange(x, "b im t k c -> (b t) (im k) c")
                pos_cross = rearrange(pos, "b im t k d -> (b t) (im k) d") if pos is not None else None

                # im = t
                mem_cross = rearrange(memory, "b im l c -> (b im) l c")
                mem_pos_cross = rearrange(memory_pos, "b im l d -> (b im) l d") if memory_pos is not None else None
            else:
                x_cross = rearrange(x, "b im t k c -> b (im t k) c")
                pos_cross = rearrange(pos, "b im t k d -> b (im t k) d") if pos is not None else None

                mem_cross = rearrange(memory, "b im l c -> b (im l) c")
                mem_pos_cross = rearrange(memory_pos, "b im l d -> b (im l) d") if memory_pos is not None else None


            # Residual + Norm
            shortcut = x_cross
            x_cross = self.norm2(x_cross)

            # Apply Cross Attention
            x_cross = self.attn_cross(
                x=x_cross, 
                memory=mem_cross, 
                pos=pos_cross, 
                pos_memory=mem_pos_cross, 
            )
            
            x_cross = self.ls2(x_cross)
            x_cross = shortcut + self.drop_path2(x_cross)

            # reshape
            if use_mask_attn:
                x = rearrange(x_cross, "(b t) (im k) c -> b im t k c", b=b, im=im, k=k)
            else:
                x = rearrange(x_cross, "b (im t k) c -> b im t k c", im=im, t=t, k=k)

        # --- 3. FFN ---
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))

        return x

        
class SpatialTemporalMod(nn.Module):
    def __init__(
        self,
        # input_feature
        feature_dim=2048,
        embed_dim=1024,
        feature_level=4,
        patch_size=14,
        detach_pts3d=False,
        detach_feats=False,
        # cross_attention_blocks
        num_heads=16,
        mlp_ratio=4.0,
        num_block=16,
        intermediate_layer_idx=[4, 11, 17, 23],
        block_fn=CrossAttnBlock,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        init_values=0.01,
        rope_freq=100,
        # dynamic_token,
        resize_factor=1,
        dynamic_topk=0.1,
        weight_dynamic_score=True,
        weight_dynamic_score_detach=True,
        weight_dynamic_score_alpha=0.3,
        # time_embed
        use_time_embed=True,
        scale_time_embed=True,
        scale_time_embed_by_sp_adaln=False,
        scale_time_embed_by_adaln=False,
        use_time_embed_feat=True,
        # motion_field
        motion_dim=9,
        spatial_temporal_feature_idx=[7, 15],
        # gs_field
        use_gs_field=False,
        gs_dim=16,
        gs_feature_idx=[7, 15],
        # misc
        temporal_attn_layer_idx=[],
        spatial_attn_layer_idx=[],
        use_ref_temporal_attn=False,
        cross_attn_layer_idx=[],
        mask_attn_layer_idx=[],
        use_checkpointing=False,
    ):

        super().__init__()

        self.patch_size = patch_size
        self.feature_proj = nn.Linear(feature_dim*feature_level, embed_dim)
        # TODO: might use sinusoid embedding here, or other mod
        self.pts3d_proj = nn.Linear(patch_size*patch_size*3, embed_dim)

        self.detach_pts3d = detach_pts3d
        self.detach_feats = detach_feats
        self.embed_dim = embed_dim

        self.position_getter = PositionGetter()
        # cross-attn
        rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=rope,
                    use_temporal_attn=(block_idx in temporal_attn_layer_idx),
                    use_spatial_attn=(block_idx in spatial_attn_layer_idx),
                    use_ref_temporal_attn=use_ref_temporal_attn, # tmp_debug
                    use_cross_attn=(block_idx in cross_attn_layer_idx),
                    attn1_class=AttentionMask if not use_ref_temporal_attn else CrossAttentionMask
                )
                    for block_idx in range(num_block)
            ]
        )
        self.num_block = num_block

        self.resize_factor = resize_factor
        if resize_factor > 1:
            self.resize_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=embed_dim, 
                    out_channels=embed_dim * (resize_factor ** 2), 
                    kernel_size=1,  # 1x1 卷积，不看邻居，只看自己，符合你"Patch内分裂"的需求
                    padding=0, 
                    bias=True
                ),
                nn.PixelShuffle(upscale_factor=resize_factor)
            )

        hidden_dim = embed_dim // 2
        mid_dim = max(32, embed_dim // 4)
        self.dynamic_pred = nn.Sequential(
            # nn.Linear(embed_dim, hidden_dim) -> nn.Conv2d
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),  # 或 GELU
            nn.Dropout(0.1),
            
            # nn.Linear(hidden_dim, mid_dim) -> nn.Conv2d
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 最后一层也改为 Conv2d 1x1
        self.dynamic_pred_last_layer = nn.Conv2d(mid_dim, 1, kernel_size=1)

        self.dynamic_topk = dynamic_topk
        self.weight_dynamic_score = weight_dynamic_score
        self.weight_dynamic_score_detach = weight_dynamic_score_detach
        self.weight_dynamic_score_alpha = weight_dynamic_score_alpha

        self.use_time_embed_feat = use_time_embed_feat
        self.scale_time_embed = scale_time_embed
        self.scale_time_embed_by_sp_adaln = scale_time_embed_by_sp_adaln
        self.scale_time_embed_by_adaln = scale_time_embed_by_adaln

        assert not (scale_time_embed_by_sp_adaln and scale_time_embed_by_adaln), "only support one adaln is true"

        self.use_time_embed = use_time_embed

        self.temporal_attn_layer_idx = temporal_attn_layer_idx
        self.spatial_attn_layer_idx = spatial_attn_layer_idx
        self.cross_attn_layer_idx = cross_attn_layer_idx
        self.mask_attn_layer_idx = mask_attn_layer_idx
        
        self.use_checkpointing = use_checkpointing

        # time_embedding, 
        # shared by feat and query
        if self.use_time_embed:
            self.video_idx_embed = SinusoidPosEmbed(embed_dim=embed_dim, use_proj=True, init_zero=True)
            self.local_time_idx_embed = SinusoidPosEmbed(embed_dim=embed_dim, use_proj=True, init_zero=True)
     
        if self.use_time_embed:
            if self.scale_time_embed_by_sp_adaln:
                self.sp_adaln = SpatialTemporalAdaLN(dim=embed_dim)
            elif self.scale_time_embed_by_adaln:
                self.sp_adaln = TemporalAdaLN(dim=embed_dim)
            elif self.scale_time_embed:
                # self.time_embed_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

                hidden_dim = embed_dim // 2  
                mid_dim = max(32, embed_dim // 4) 
                self.time_embed_scaling_mlp = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),  # 或 GELU
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, mid_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(mid_dim, 1),
                    nn.Softplus()  # 输出范围 >=0，无上限
                )

        # motion
        self.spatial_temporal_feature_idx = spatial_temporal_feature_idx
        self.motion_proj_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),  # 或 GELU
                nn.Linear(hidden_dim, motion_dim)  # 6drot + 3dtrans
            ) 
            for _ in range(len(self.spatial_temporal_feature_idx))
        ])
        self.motion_dim = motion_dim
        # reset param
        if motion_dim == 9 or motion_dim == 3:
            for proj in self.motion_proj_list:
                torch.nn.init.zeros_(proj[-1].weight)
        
        # gs
        self.use_gs_field = use_gs_field
        if self.use_gs_field:
            self.gs_proj_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),  # 或 GELU
                    nn.Linear(hidden_dim, gs_dim)  # 6drot + 3dtrans
                ) 
                for _ in range(len(self.gs_feature_idx))
            ])
            self.gs_video_idx_embed = SinusoidPosEmbed(embed_dim=gs_dim, use_proj=True, init_zero=True)
            self.gs_local_time_idx_embed = SinusoidPosEmbed(embed_dim=gs_dim, use_proj=True, init_zero=True)

    def forward(self, feats, output, image_info=None):
        """
            Args:
                feats: list[tuple(output_feature, camera_token)]
                    output_feature: [b im l c]
                    camera_token: [b im c]
                output: Dict
                    depth: [b im h w]
                    depth_conf: [b im h w]
                    extrinsics (w2c): [b im 3 4]
                    intrinsics: [b im 3 3]
                    pts3d: [b im h w 3]
                    
        """
        # TODO support resize_factor 1x -> 2x
        # 0. preparation: time_embedding and collect feat
        b, im, h, w = output["depth"].shape
        t = im # one image per time
        
        hp, wp = h // self.patch_size, w // self.patch_size
        if self.use_time_embed:
            video_idx_embedding = self.video_idx_embed(image_info[:, :, 1]) # [b im c]
            local_time_idx_embedding = self.local_time_idx_embed(image_info[:, :, 2]) # [b im c]
        
        feats = [feat[0] for feat in feats]

        # 1. pts3d_embedding
        pts3d = output["pts3d"]
        if self.detach_pts3d:
            pts3d = pts3d.clone().detach()
        if self.detach_feats:
            feats = [feat.clone().detach() for feat in feats]

        pts3d_reshaped = rearrange(pts3d, "b im (hp p1) (wp p2) xyz -> b im (hp wp) (p1 p2 xyz)", p1=self.patch_size, p2=self.patch_size)
        pts3d_embedding = self.pts3d_proj(pts3d_reshaped)

        # 2. feats_embedding
        feats_embedding = self.feature_proj(torch.cat(feats, dim=-1)) # [b im l c]
        if self.use_time_embed_feat:
            feats_embedding = feats_embedding + video_idx_embedding.unsqueeze(2) + local_time_idx_embedding.unsqueeze(2)
        
        # value
        feats_embedding = feats_embedding + pts3d_embedding
        feats_pos = self.position_getter(
            b*im, hp, wp, device=feats_embedding.device
        )
        feats_pos = feats_pos.reshape(b, im, hp*wp, -1)

        # 3. dynamic_query [per image]
        # expand dynamic_query by 2x if need
        dynamic_feature = rearrange(feats_embedding, "b im (hp wp) d -> (b im) d hp wp", hp=hp, wp=wp)
        # resize factor defined initialization
        if self.resize_factor > 1:
            dynamic_feature = self.resize_layer(dynamic_feature)
        resized_hp, resized_wp = dynamic_feature.shape[-2:]
        # pos for resize dynamic_feature
        dynamic_feature_pos = self.position_getter(
            b*im, resized_hp, resized_wp, device=feats_embedding.device
        )
        dynamic_feature_pos = dynamic_feature_pos.reshape(b, im, resized_hp*resized_wp, -1)


        dynamic_score_logits_before_last = self.dynamic_pred(dynamic_feature)
        dynamic_score_logits = self.dynamic_pred_last_layer(dynamic_score_logits_before_last)
        dynamic_score = F.sigmoid(dynamic_score_logits).squeeze(1)
        if self.weight_dynamic_score:
            if self.weight_dynamic_score_detach:
                dynamic_score_logits_before_last = dynamic_score_logits_before_last.clone().detach()
            dy_similarity = dwconv_avg8_cosine(dynamic_score_logits_before_last)
            scale = (1 - dy_similarity * self.weight_dynamic_score_alpha)
            dynamic_score = dynamic_score * scale
        dynamic_score = rearrange(dynamic_score, "(b im) hp wp -> b im (hp wp)", b=b, im=im)


        k = max(1, int(resized_hp*resized_wp*self.dynamic_topk))  # topk dynamic token
        _, topk_indices = torch.topk(dynamic_score, k=k, dim=2)  # [b, im, k]
        dynamic_mask = torch.zeros(b, im, resized_hp*resized_wp, dtype=torch.bool, device=feats_embedding.device)  # [b, im, l]
        dynamic_mask.scatter_(2, topk_indices, True)
        dynamic_mask = dynamic_mask.float()
        # query
        # dynamic_token = dynamic_feature[dynamic_mask] # [b im k c]
        dynamic_feature = rearrange(dynamic_feature, "(b im) c hp wp -> b im (hp wp) c", im=im, hp=resized_hp, wp=resized_wp)
        indices_for_gather = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        # dynamic_token: [b, im, k, c]
        dynamic_token = torch.gather(dynamic_feature, 2, indices_for_gather) 
        
        if self.use_time_embed:
            if self.scale_time_embed_by_sp_adaln or self.scale_time_embed_by_adaln:
                dynamic_token = self.sp_adaln(dynamic_token, (video_idx_embedding+local_time_idx_embedding))
            else:
                if self.scale_time_embed:
                    time_embed_scaling = self.time_embed_scaling_mlp(dynamic_token) # [b im k 1]
                    # dynamic_token = self.time_embed_norm(dynamic_token)
                else:
                    time_embed_scaling = torch.ones(b, im, k, 1).to(dynamic_token)
                # TODO: use expaned_video_idx and local_time_idx
                dynamic_token = dynamic_token.unsqueeze(2) + time_embed_scaling.unsqueeze(2) * (video_idx_embedding[:, None, :, None] \
                    + local_time_idx_embedding[:, None, :, None]) # [b im t k c]
        else:
            dynamic_token = dynamic_token.unsqueeze(2).repeat(1, 1, t, 1, 1)

        indices_for_gather_pos = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        dynamic_token_pos = torch.gather(dynamic_feature_pos, 2, indices_for_gather_pos)
        dynamic_token_pos = dynamic_token_pos[:,:, None].repeat(1, 1, t, 1, 1)

        out_dynamic_token_list = []
        # 5. cross-attention block between dynamic_query and feats (we use list here)
        for block_idx in range(self.num_block):
            if self.training and self.use_checkpointing:
                dynamic_token = checkpoint(
                    self.blocks[block_idx],
                    dynamic_token,
                    feats_embedding,
                    dynamic_token_pos,
                    feats_pos,
                    (block_idx in self.mask_attn_layer_idx),
                    (block_idx in self.temporal_attn_layer_idx),
                    (block_idx in self.spatial_attn_layer_idx),
                    (block_idx in self.cross_attn_layer_idx),
                    use_reentrant=False,
                )
            else:
                dynamic_token = self.blocks[block_idx](
                    dynamic_token, feats_embedding, dynamic_token_pos, feats_pos, use_temporal_attn=(block_idx in self.temporal_attn_layer_idx),
                    use_mask_attn=(block_idx in self.mask_attn_layer_idx), use_spatial_attn=(block_idx in self.spatial_attn_layer_idx),
                    use_cross_attn=(block_idx in self.cross_attn_layer_idx),
                )
            if block_idx in self.spatial_temporal_feature_idx:
                out_dynamic_token_list.append(dynamic_token)

        # 6. pred motion field, record motion feature

        # output_feature_list = []
        motion_field_list = []
        if self.use_gs_field:
            gs_field_list = []
            gs_video_idx_embedding = self.video_idx_embed(image_info[:, :, 1]) # [b im c]
            gs_local_time_idx_embedding = self.local_time_idx_embed(image_info[:, :, 2]) # [b im c]

        for idx, dynamic_token in enumerate(out_dynamic_token_list):
            out_feature = torch.zeros(b, im, t, resized_hp*resized_wp, self.embed_dim).to(dynamic_token)
            indices_feat = topk_indices.unsqueeze(2).unsqueeze(-1) # [b, im, 1, k, 1]
            indices_feat = indices_feat.expand(b, im, t, k, self.embed_dim) # [b, im, t, k, c]

            out_feature.scatter_(dim=3, index=indices_feat, src=dynamic_token)
            # output_feature_list.append(out_feature)

            # motion
            if self.motion_dim == 9:
                output_motion = torch.zeros(b, im, t, resized_hp*resized_wp, self.motion_dim).to(dynamic_token)
                output_motion[..., 0] = 0.001
                output_motion[..., 4] = 0.001
            else:
                output_motion = self.motion_proj_list[idx](dynamic_feature)
                output_motion = output_motion[:,:, None].repeat(1, 1, t, 1, 1).to(dynamic_token)

            indices_motion = topk_indices.unsqueeze(2).unsqueeze(-1) # [b, im, 1, k, 1]
            indices_motion = indices_motion.expand(b, im, t, k, self.motion_dim) # [b, im, t, k, c]
            
            motion_vec = self.motion_proj_list[idx](dynamic_token).float() # tmp_debug
            output_motion.scatter_(dim=3, index=indices_motion, src=motion_vec)

            motion_field_list.append(output_motion)


            if self.use_gs_field:
                gs_field = self.gs_field_proj_list[idx](dynamic_feature)
                gs_field = gs_field[:,:, None].repeat(1, 1, t, 1, 1).to(dynamic_token)
                
                gs_vec = self.gs_field_proj_list[idx](dynamic_token).float() # tmp_debug
                gs_field.scatter_(dim=3, index=indices_motion, src=gs_vec)
                gs_field = gs_field + gs_video_idx_embedding[:, None, :, None] + gs_local_time_idx_embedding[:, None, :, None]

                gs_field_list.append(gs_field)

              


        dynamic_token_info = {
            "dynamic_score": dynamic_score,
            "dynamic_mask": dynamic_mask - dynamic_score.clone().detach() + dynamic_score,
        }
        output = {
            # "output_feature_list": output_feature_list,
            "motion_field_list": motion_field_list,
            "dynamic_token_info": dynamic_token_info,
        }

        if self.use_gs_field:
            output["gs_field_list"] = gs_field_list

        return output


def dwconv_avg8_cosine(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    用3x3 depthwise conv 计算每个位置与其8邻域的平均余弦相似度（zero padding）。
    输入: f [B, C, H, W]
    输出: sim [B, H, W]
    """
    # 通道维做L2归一化，便于用点积表示cosine
    f_norm = F.normalize(f, dim=1, eps=eps)
    # 3x3核：中心为0，其余8个位置为1/8；对每个通道单独卷积（depthwise）
    B, C, H, W = f.shape
    mask = f.new_ones(3, 3)
    mask[1, 1] = 0.0
    weight = (mask / 8.0).view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # [C,1,3,3]

    # 零填充 + depthwise conv 得到邻域均值向量
    mean_vec = F.conv2d(f_norm, weight, padding=1, groups=C)  # [B,C,H,W]

    # 平均余弦相似度 = 单位向量 · 邻域均值向量
    sim = (f_norm * mean_vec).sum(dim=1)  # [B,H,W]
    return sim


class SpatialTemporalAdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # 0. 加入 LayerNorm
        # 设置 elementwise_affine=False，因为缩放由后面的 dy 和 t 参数共同决定
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # 1. Dynamic Token 投影层 (生成空间维度的 scale & shift)
        # 输入 dim, 输出 2 (每个 token 一个标量 scale 和一个标量 shift)
        self.dy_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, 2)
        )
        
        # 2. Time Embedding 投影层 (生成时间维度的 scale & shift)
        # 输入 dim, 输出 2 * dim (每个 channel 一个 scale 和一个 shift)
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
        
        # 3. Zero-Initialization (AdaLN-Zero 策略)
        # 初始化为 0，使训练初期输出等于 LayerNorm 后的结果在时间轴上的复制
        nn.init.constant_(self.dy_proj[1].weight, 0)
        nn.init.constant_(self.dy_proj[1].bias, 0)
        
        nn.init.constant_(self.t_proj[1].weight, 0)
        nn.init.constant_(self.t_proj[1].bias, 0)

    def forward(self, dynamic_token, time_embedding):
        """
        Args:
            dynamic_token:  [B, IM, K, C] (初始状态特征)
            time_embedding: [B, T, C]    (时间步特征)
        Returns:
            out:            [B, IM, T, K, C]
        """
        
        # ==========================================
        # Part A: 标准化 (Normalization)
        # ==========================================
        # 对原始 token 进行归一化，这是 AdaLN 的核心步骤
        # x_norm: [B, IM, K, C]
        x_norm = self.norm(dynamic_token)

        # ==========================================
        # Part B: 生成空间参数 (Spatial Params)
        # ==========================================
        # 从原始 dynamic_token 中提取空间特有的调制信号
        # dy_params: [B, IM, K, 2]
        dy_params = self.dy_proj(dynamic_token)
        scale_dy, shift_dy = dy_params.chunk(2, dim=-1) # [B, IM, K, 1]

        # ==========================================
        # Part C: 生成时间参数 (Temporal Params)
        # ==========================================
        # 从 time_embedding 中提取时间特有的调制信号
        # t_params: [B, T, 2*C]
        t_params = self.t_proj(time_embedding)
        scale_t, shift_t = t_params.chunk(2, dim=-1) # [B, T, C]

        # ==========================================
        # Part D: 维度扩充与广播
        # ==========================================
        # 目标形状: [B, IM, T, K, C]
        
        # 1. 归一化后的数据扩充 T 轴
        # [B, IM, K, C] -> [B, IM, 1, K, C]
        x = x_norm.unsqueeze(2)
        
        # 2. 空间参数扩充 T 轴和 C 轴 (因为是标量，C轴保持1)
        # [B, IM, K, 1] -> [B, IM, 1, K, 1]
        s_dy = scale_dy.unsqueeze(2)
        b_dy = shift_dy.unsqueeze(2)
        
        # 3. 时间参数扩充 IM 轴和 K 轴
        # [B, T, C] -> [B, 1, T, 1, C]
        s_t = scale_t.unsqueeze(1).unsqueeze(3)
        b_t = shift_t.unsqueeze(1).unsqueeze(3)

        # ==========================================
        # Part E: 混合调制 (The Core Formula)
        # ==========================================
        # 公式: x_norm * (1 + scale_dy * scale_t) + (shift_dy * shift_t)
        
        combined_scale = s_dy * s_t  # 空间标量与时间向量相乘
        combined_shift = b_dy * b_t  # 空间偏移与时间偏移相乘
        
        # 应用调制
        out = x * (1 + combined_scale) + combined_shift
        
        return out


class TemporalAdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # 1. 对输入特征进行归一化
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # 2. 从时间信息中提取调制参数
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim) # 只需要 scale 和 shift
        )
        
        # 3. 初始化为 0
        # 这样初始状态下 scale=0, shift=0，输出就是 x_norm 在时间轴上的重复
        nn.init.constant_(self.t_proj[1].weight, 0)
        nn.init.constant_(self.t_proj[1].bias, 0)

    def forward(self, dynamic_token, time_embedding):
        """
        Args:
            dynamic_token:  [B, IM, K, C] (比如初始状态)
            time_embedding: [B, T, C]    (比如预测的时间步 embedding)
        Returns:
            out:            [B, IM, T, K, C]
        """
        # Step A: 先对 token 归一化
        # [B, IM, K, C]
        x_norm = self.norm(dynamic_token)
        
        # Step B: 扩展 x 的维度以准备在时间轴 T 上广播
        # [B, IM, K, C] -> [B, IM, 1, K, C]
        x = x_norm.unsqueeze(2)
        
        # Step C: 生成时间调制参数
        # time_embedding: [B, T, C] -> [B, T, 2*C]
        t_params = self.t_proj(time_embedding)
        scale_t, shift_t = t_params.chunk(2, dim=-1) # 各自为 [B, T, C]

        # Step D: 调整参数维度以便广播到 [B, IM, T, K, C]
        # [B, T, C] -> [B, 1, T, 1, C]
        s_t = scale_t.unsqueeze(1).unsqueeze(3)
        b_t = shift_t.unsqueeze(1).unsqueeze(3)

        # Step E: 执行调制
        # 即使没有 Residual，这里的 (1 + s_t) 也是核心逻辑
        # 初始时 s_t=0, b_t=0, 则 out 就是 x 在 T 轴上的 repeat
        out = x * (1 + s_t) + b_t
        
        return out