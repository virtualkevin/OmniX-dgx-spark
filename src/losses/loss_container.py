from copy import copy, deepcopy

import torch
import torch.nn as nn
import math
    

class BaseLoss(nn.Module):
    """
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def __repr__(self):
        name = self.name
        return name

    def forward(self, *args, **kwargs):
        loss, details = self.compute_loss(*args, **kwargs)
        return loss, details


class LossContainer(BaseLoss):
    """
    容器类，用于组合多个损失函数
    """
    def __init__(self, losses=None, weights=None):
        super().__init__()

        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
        if len(self.weights) != len(self.losses):
            raise ValueError(f"权重数量 ({len(self.weights)}) 必须等于损失函数数量 ({len(self.losses)})")
    
    def compute_loss(self, gts, preds, *args, **kwargs):
        total_loss = 0
        combined_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            
            loss, loss_dict = loss_fn(preds, gts, *args, **kwargs)
            total_loss = total_loss + weight * loss
            for key, value in loss_dict.items():
                combined_dict[f"{key}"] = value
        combined_dict["total_loss"] = total_loss

        # TODO: we do not have this now, but we will have it in the future
        if "bad_case" in gts and gts["bad_case"]:
            total_loss, combined_dict = self.skip_bad_case(total_loss, combined_dict)
        
        return total_loss, combined_dict

    def get_name(self):
        return "CombinedLoss"

    def skip_bad_case(self, total_loss, loss_dict):
        for _ in range(10):
            print("Bad Case! Skip loss!")
        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(0.0, device=total_loss.device),
            total_loss
            ) * 0
        for key, val in list(loss_dict.items()):
            if torch.is_tensor(val):
                # 浮点/复数张量先去掉 NaN/Inf，再置零；非浮点（如 long/bool）直接置零
                if val.is_floating_point() or val.is_complex():
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                loss_dict[key] = torch.zeros_like(val)
            elif isinstance(val, (float, int)):
                # Python 数值类型：非有限值先置 0，再清零
                if not math.isfinite(float(val)):
                    val = 0.0
                loss_dict[key] = 0.0
            else:
                # 其它类型（比如字符串、dict），不参与反传/日志可保留或按需处理
                # combined_dict[key] = val
                pass
        return total_loss, loss_dict