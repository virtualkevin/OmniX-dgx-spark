import torch
import torch.nn as nn

class SinusoidPosEmbed(nn.Module):
    """
    Batch支持的 1D sinusoidal positional embedding module.
    支持输入 [B, N]，为每个位置生成 sine+cos 的embedding。
    """
    
    def __init__(self, embed_dim: int, omega_0: float = 100, use_proj: bool = False, init_zero: bool = False):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        
        self.embed_dim = embed_dim
        self.omega_0 = omega_0
        self.use_proj = use_proj
                
        if use_proj:
            self.proj = nn.Linear(embed_dim, embed_dim)
            if init_zero:
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: Position tensor of shape [B, N] (float or int)
        
        Returns:
            emb: [B, N, embed_dim] positional embeddings
        """
        device = pos.device
        
        # TODO: check dtype of omega
        # [D/2]
        omega = torch.arange(
            self.embed_dim // 2, 
            dtype=torch.float32, 
            device=device
        )
        omega = omega / (self.embed_dim / 2.0)
        omega = 1.0 / (self.omega_0 ** omega)  # (D/2,)
        
        # 计算外积 [B, N, D/2]
        out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)  # broadcasting
        
        # sin / cos
        emb_sin = torch.sin(out)  # [B, N, D/2]
        emb_cos = torch.cos(out)  # [B, N, D/2]
        
        # 拼接 --> [B, N, D]
        emb = torch.cat([emb_sin, emb_cos], dim=-1)

        if self.use_proj:
            emb = self.proj(emb)  # 线性层会自动处理最后一维
        
        return emb.float()