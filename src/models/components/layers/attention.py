import torch

from flash_attn_interface import flash_attn_func as flash_attn_fuc_v3
from depth_anything_3.model.dinov2.layers.attention import Attention

from torch import Tensor
import torch.nn.functional as F


class AttentionFA3(Attention):

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if q.dtype == torch.bfloat16 or q.dtype == torch.float16:
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
            # print("not use fa3")          
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    attn_mask=(
                        (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                        if attn_mask is not None
                        else None
                    ),
                )
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