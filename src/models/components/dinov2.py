import torch.nn as nn

from typing import List
from src.models.components.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

class CustomDinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        use_time_embed: bool = False,
        time_embed_start_idx: int = 13,
        freeze_backbone_layers: int = -1,
        checkpointing_prepare: bool = False,
        checkpointing_start_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            use_time_embed=use_time_embed,
            time_embed_start_idx=time_embed_start_idx,
            freeze_backbone_layers=freeze_backbone_layers,
            checkpointing_prepare=checkpointing_prepare,
            checkpointing_start_idx=checkpointing_start_idx,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
