import contextlib
import torch
import torch.nn as nn

from src.models.components.embedding import SinusoidPosEmbed
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
from depth_anything_3.model.reference_view_selector import (
    RefViewStrategy,
    select_reference_view,
    reorder_by_reference,
    restore_original_order,
)
from depth_anything_3.model.dinov2.layers import (  # noqa: F401
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)

from depth_anything_3.utils.logger import logger
from src.models.components.layers.attention import AttentionFA3
import torch.utils.checkpoint as cp

# TODO:
#   add checkpointing when training, support _get_intermediate_layers_chunked
class CustomDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
        # custom
        use_time_embed=False, 
        time_embed_start_idx=13, 
        freeze_backbone_layers=-1,
        checkpointing_prepare=False, 
        checkpointing_start_idx=None
    ):
    
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        # modify - AttentionFA3
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
                # modify
                attn_class=AttentionFA3,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        # custom
        self.use_time_embed = use_time_embed
        self.time_embed_start_idx = time_embed_start_idx
        self.freeze_backbone_layers = freeze_backbone_layers

        self.checkpointing_prepare = checkpointing_prepare
        self.checkpointing_start_idx = checkpointing_start_idx

        assert time_embed_start_idx >= freeze_backbone_layers, f"time_embed should >= freeze_backbone_layers"
        # time_embedding
        if use_time_embed:
            self.video_idx_embed = SinusoidPosEmbed(embed_dim=embed_dim, use_proj=True, init_zero=True)
            self.local_time_idx_embed = SinusoidPosEmbed(embed_dim=embed_dim, use_proj=True, init_zero=True)
    
    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], image_info=None, **kwargs):
        
        # --- 获取 Checkpoint 配置 ---
        # 使用 getattr 以兼容可能没有在 __init__ 中定义这些属性的情况
        # checkpointing_start_idx: 从第几层开始 checkpoint (None 表示不开启)
        # checkpointing_prepare: 是否 checkpoint 预处理阶段
        ckpt_start_idx = getattr(self, "checkpointing_start_idx", None)
        ckpt_prepare = getattr(self, "checkpointing_prepare", False)

        B, S, _, H, W = x.shape

        # ---------------- 1. Prepare Tokens 的 Checkpoint 逻辑 ----------------
        use_ckpt_prepare = ckpt_prepare and (self.freeze_backbone_layers == -1)
        if use_ckpt_prepare:
            x = cp.checkpoint(self.prepare_tokens_with_masks, x, use_reentrant=False)
        elif self.freeze_backbone_layers != -1:
            with torch.no_grad():
                x = self.prepare_tokens_with_masks(x)
        else:
            x = self.prepare_tokens_with_masks(x)

        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
        
        for i, blk in enumerate(self.blocks):
            
            # 1. 动态决定当前层是否需要梯度
            is_frozen = (self.freeze_backbone_layers != -1) and (i < self.freeze_backbone_layers)
            ctx = torch.no_grad() if is_frozen else contextlib.nullcontext()

            # 2. 处理冻结交界处
            if i == self.freeze_backbone_layers:
                x = x.detach()
                x.requires_grad_(True)

            with ctx:
                # RoPE 准备
                if i < self.rope_start or self.rope is None:
                    g_pos, l_pos = None, None
                else:
                    g_pos = pos_nodiff
                    l_pos = pos

                # cam_token 修改
                if self.alt_start != -1 and i == self.alt_start:
                    if kwargs.get("cam_token", None) is not None:
                        logger.info("Using camera conditions provided by the user")
                        cam_token = kwargs.get("cam_token")
                    else:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                    x = torch.cat([cam_token[:, :, None], x[:, :, 1:]], dim=2)
                    # x[:, :, 0] = cam_token

                # time_embed
                if self.use_time_embed and i == self.time_embed_start_idx:
                    video_idx_embedding = self.video_idx_embed(image_info[:, :, 1])
                    local_time_idx_embedding = self.local_time_idx_embed(image_info[:, :, 2])
                    x = x + video_idx_embedding.unsqueeze(2).to(x) + local_time_idx_embedding.unsqueeze(2).to(x)
                
                # ---------------- 2. Attention 的 Checkpoint 逻辑 ----------------
                is_global = (self.alt_start != -1 and i >= self.alt_start and i % 2 == 1)
                mode = "global" if is_global else "local"
                curr_pos = g_pos if is_global else l_pos
                curr_mask = kwargs.get("attn_mask", None) if is_global else None
                
                # 判断是否启用 Checkpoint:
                # 1. 设置了 start_idx 且当前层数达到要求
                # 2. 且当前层未被冻结
                do_checkpoint = (
                    ckpt_start_idx is not None 
                    and i >= ckpt_start_idx 
                    and not is_frozen
                )

                if do_checkpoint:
                    def run_blk_wrapper(x_, pos_, mask_, b=blk, m=mode):
                        return self.process_attention(x_, b, m, pos=pos_, attn_mask=mask_)
                    
                    x = cp.checkpoint(run_blk_wrapper, x, curr_pos, curr_mask, use_reentrant=False)
                else:
                    x = self.process_attention(x, blk, mode, pos=curr_pos, attn_mask=curr_mask)
    
                if not is_global:
                    local_x = x
                
                if i in blocks_to_take:
                    out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                    output.append((out_x[:, :, 0], out_x))
                if i in export_feat_layers:
                    aux_output.append(x)

        return output, aux_output


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = CustomDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = CustomDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = CustomDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = CustomDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model