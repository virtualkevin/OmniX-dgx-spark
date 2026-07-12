from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from src.models.components.dinov2 import CustomDinoV2
from depth_anything_3.model.cam_dec import CameraDec
from depth_anything_3.model.dualdpt import DualDPT

from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

from src.models.components.st_module import SpatialTemporalMod
from src.models.components.trajectory import TrajectoryMod
from src.models.components.offset_head import OffsetDPTHead

from src.utils.projection import depthmap_to_world_coordinates
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

# gaussian
# import render_3dgs
# from depth_anything_3.model.gsdpt import GSDPT
# from src.models.components.gs_adapter import GaussianAdapter

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# fix exp
def patched_apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
    """
    完全自包含的补丁函数，增加了数值安全保护（Safety Clamps）。
    防止 exp 导致 inf (上溢) 或 0 (下溢)。
    """
    act = activation.lower() if isinstance(activation, str) else activation
    
    # --- 数值保护阈值 ---
    # float32 的 exp(88.7) 会变成 inf，我们限制到 60 (约 10^26)，足够覆盖物理深度范围
    # 限制下限到 -60 (约 10^-26)，防止变成绝对的 0 导致除零错误
    SAFE_UPPER = 10.0
    SAFE_LOWER = -10.0

    if act == "exp":
        return torch.exp(torch.clamp(x, min=SAFE_LOWER, max=SAFE_UPPER))
    
    if act == "expm1":
        return torch.expm1(torch.clamp(x, min=SAFE_LOWER, max=SAFE_UPPER))
    
    if act == "expp1":
        # 置信度分支通常用这个，clamp 之后即便 x 极大，结果也是有限数
        return torch.exp(torch.clamp(x, min=SAFE_LOWER, max=SAFE_UPPER)) + 1
    
    if act == "relu":
        return torch.relu(x)
    
    if act == "sigmoid":
        # Sigmoid 内部有 exp(-x)，虽然自带稳定性，但 clamp x 可以防止梯度过早消失
        return torch.sigmoid(torch.clamp(x, min=SAFE_LOWER, max=SAFE_UPPER))
    
    if act == "softplus":
        # Softplus 内部是 log(1 + exp(x))，对大值有很好的处理，但 clamp 仍能增加保险
        return F.softplus(torch.clamp(x, max=SAFE_UPPER))
    
    if act == "tanh":
        return torch.tanh(x)
    
    # 默认线性输出
    return x

# --- 执行猴子补丁 ---
# 这一步会直接修改 DualDPT 类的行为，无需改动源码文件
DualDPT._apply_activation_single = patched_apply_activation_single

print("✅ [Gemini 3 Pro] DualDPT activation safety patch applied successfully.")

# for debug
def log_gpu_mem(module_name):
    torch.cuda.synchronize()  # 保证显存数据准确
    current_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[{module_name}] Current: {current_mem/1024**2:.2f} MB | Peak: {peak_mem/1024**2:.2f} MB")

# depth anything 3
class CustomModel(nn.Module):
    """Modify the VGGT model to support custom heads and aggregated_feature_adapter

        we will add feature_interation_neck for aggregated_feature post-processing
    """
    def __init__(
        self,
        backbone_args: Dict,
        head_args: Dict,
        cam_args: Dict,
        st_mod_args: Dict = None,
        offset_head_args: Dict = None,
        trajectory_mod_args: Dict = None,
        gs_processer_args: Dict = None,
        pts3d_mode: str = "ray",
        pretrain: bool = False,
        use_ray_pose: bool = False,
    ):
        super().__init__()


        self.backbone = CustomDinoV2(**backbone_args)
        self.head = DualDPT(**head_args)
        self.cam_dec = CameraDec(**cam_args["cam_dec"])

        self.pretrain = pretrain
        
        self.use_ray_pose = use_ray_pose

        if not self.pretrain:
            # custom modules
            self.st_mod = SpatialTemporalMod(**st_mod_args)

            self.trajectory_mod = TrajectoryMod(**trajectory_mod_args)

            # build heads
            self.offset_head = OffsetDPTHead(**offset_head_args)
              
        # TODO: build gs_processer here
        enable_gs = (gs_processer_args is not None)
        self.enable_gs = enable_gs
        if self.enable_gs:
            self.gs_head = GSDPT(**gs_processer_args["gs_head_args"])
            self.gs_adapter = CustomGaussianAdapter(**gs_processer_args["gs_adapter_args"])
            self.gs_offset_head = OffsetDPTHead(**gs_processer_args["gs_offset_head_args"])
            self.gs_trajectory_mod = GaussianTrajectoryMod(**gs_processer_args["gs_trajectory_mod_args"])

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        freeze_backbone_layers = backbone_args.get("freeze_backbone_layers", -1)
        self.freeze_backbone_layers = freeze_backbone_layers
        self.pts3d_mode = pts3d_mode
        # TODO: suuport freeze certain layers
        # self.net.backbone.pretrained.patch_embed
        # self.net.backbone.pretrained.blocks (0-12) (13-39) i%2 == 1 global
        if self.freeze_backbone_layers > 0:
            
            backbone_model = self.backbone.pretrained 
            
            # 1. 冻结 Patch Embedding (通常这是最底层的特征提取)
            if hasattr(backbone_model, 'patch_embed'):
                for param in backbone_model.patch_embed.parameters():
                    param.requires_grad = False
                print("Frozen backbone: patch_embed")

            # 2. 冻结前 N 个 Blocks
            if hasattr(backbone_model, 'blocks'):
                # 假设 blocks 是一个 nn.ModuleList
                total_blocks = len(backbone_model.blocks)
                
                # 防止设置的层数超过实际层数，取最小值
                num_to_freeze = min(self.freeze_backbone_layers, total_blocks)
                
                for i in range(num_to_freeze):
                    block = backbone_model.blocks[i]
                    for param in block.parameters():
                        param.requires_grad = False
                
                print(f"Frozen backbone: blocks 0 to {num_to_freeze-1}")
        
        print("❄️ 正在冻结未使用的 Aux 层参数...")

        # 你的代码逻辑只用了最后一层 (index -1)，所以我们要冻结前面所有层
        # output_conv1_aux 和 output_conv2_aux 都是 ModuleList
        aux_layers_count = len(self.head.scratch.output_conv1_aux)

        # 循环冻结前 n-1 层 (保留最后一层)
        for i in range(aux_layers_count - 1):
            # 1. 冻结 output_conv1_aux 的第 i 层
            for param in self.head.scratch.output_conv1_aux[i].parameters():
                param.requires_grad = False
                
            # 2. 冻结 output_conv2_aux 的第 i 层
            for param in self.head.scratch.output_conv2_aux[i].parameters():
                param.requires_grad = False
                
    def forward(self, batch: dict):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # TODO: 
        #   1. normalize image here, use resnet_mean and _std
        #   2. check dpt output （sky）
        images = batch["image"]
        # normalize image        
        x = (images - self._resnet_mean) / self._resnet_std

        # keep ref_view_strategy during training
        with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
            feats, _ = self.backbone(
                x, ref_view_strategy="first", image_info=batch["image_info"]
            )
        
        # feats
        # list[4]
        #   tuple (outputs, camera_tokens)
        #   [b im l c] [b im c]
        b, im, _, h, w = x.shape
        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, h, w)
            output = self._process_camera_estimation(feats, h, w, output)
     
            if self.use_ray_pose:
                output = self._process_ray_pose_estimation(output, h, w)

        if self.pts3d_mode == "ray":
            depth = output["depth"].flatten(0, 1)
            ray = output["ray"].flatten(0, 1)
            ray_reshaped = rearrange(ray, "b hr wr d -> b d hr wr")
            target_size = depth.shape[-2:]
            ray_sampled = F.interpolate(
                ray_reshaped, 
                size=target_size,
                mode='bilinear', 
                align_corners=False
            )
            ray_sampled = rearrange(ray_sampled, "b d h w -> b h w d")
            t = ray_sampled[..., 3:]  # shape: (1, 4, 378, 504, 3)
            d = ray_sampled[..., :3]  # shape: (1, 4, 378, 504, 3)
 
            pts3d = t + depth.unsqueeze(-1) * d
            
        elif self.pts3d_mode == "cam":
            pts3d = depthmap_to_world_coordinates(
                output["depth"].flatten(0, 1), 
                output["intrinsics"].flatten(0, 1), 
                output["c2w"].flatten(0, 1), 
                has_batch=True
            )
        else:
            raise NotImplementedError(f"pts3d_mode {self.pts3d_mode} is not supported yet")

        pts3d = rearrange(pts3d, "(b im) h w xyz -> b im h w xyz", im=im)
        output["pts3d"] = pts3d
        # this is a post-processing method, we remove it during training
        # output = self._process_mono_sky_estimation(output)    

        if not self.pretrain:
            # st_mod
            # might add time_embedding
            st_output = self.st_mod(feats, output, image_info=batch["image_info"])
            # offset_head
            with torch.autocast(device_type=x.device.type, enabled=False):
                offset_output = self.offset_head(
                    feats, 
                    h,
                    w,
                    patch_start_idx=0,
                    motion_field_list=st_output["motion_field_list"], 
                    dynamic_token_info=st_output["dynamic_token_info"], 
                )
            
            # trajectory_mod
            traj_output = self.trajectory_mod(
                pts3d, 
                offset_output, 
                motion_field_list=st_output["motion_field_list"],
                dynamic_token_info=st_output["dynamic_token_info"],
                time_info=batch["image_info"][..., -1],
            )

        # gs_mod
        if self.enable_gs:
            with torch.autocast(device_type=x.device.type, enabled=False):
                gs_offset_output = self.offset_head(
                    feats, 
                    h,
                    w,
                    patch_start_idx=0,
                    motion_field_list=st_output["gs_field_list"], 
                    dynamic_token_info=st_output["dynamic_token_info"], 
                )

            gs_traj_output = self.gs_trajectory_mod(
                gs_offset_output, 
                motion_field_list=st_output["gs_field_list"],
                dynamic_token_info=st_output["dynamic_token_info"],
                time_info=batch["image_info"][..., -1],
            ) # [b im t h w c]

            # TODO: check x
            # random select timestamp and render with this specific time
            random_time = torch.randint(0, im, (1,)).item()
            target_gs_traj = gs_traj_output[:, :, random_time]
            output = self._process_gs_head(feats, h, w, output, x, target_gs_traj)

            # render gs


        predictions = {}
        if not self.pretrain:
            predictions["trajectory"] = traj_output["pts3d_trajectory"]
            if "pts3d_trajectory_coarse" in traj_output:
                predictions["trajectory_coarse"] = traj_output["pts3d_trajectory_coarse"]

        predictions["depth"] = output["depth"]
        predictions["depth_conf"] = output["depth_conf"]
        predictions["ray"] = output["ray"]
        predictions["ray_conf"] = output["ray_conf"]
        predictions["pts3d"] = output["pts3d"]
        predictions["pose_enc"] = output["pose_enc"]
        predictions["intrinsics"] = output["intrinsics"]
        predictions["camera_pose"] = output["c2w"]
        if self.use_ray_pose:
            predictions["intrinsics_ray"] = output["intrinsics_ray"]
            predictions["camera_pose_ray"] = output["extrinsics_ray"]
            
        # record dynamic_score
        if not self.pretrain:
            dynamic_token_info = st_output["dynamic_token_info"]
            predictions["dynamic_token_score"] = dynamic_token_info["dynamic_score"]
            predictions["dynamic_token_mask"] = dynamic_token_info["dynamic_mask"]
            predictions["pts3d_dynamic_score"] = traj_output["pts3d_dynamic_score"]

        # offset conf
        if not self.pretrain:
            if self.offset_head.pred_conf:
                sampling_conf = offset_output["sampling_conf"]   
                # if self.pts3d_mode == "ray":
                #     ray_conf = output["ray_conf"]
                #     ray_conf_sampled = F.interpolate(
                #         ray_conf,
                #         size=(h, w),
                #         mode="bilinear",
                #         align_corners=False
                #     )
                #     depth_conf = output["depth_conf"]
                #     all_conf = torch.stack([depth_conf, ray_conf_sampled, sampling_conf], dim=-1)
                #     pts3d_conf, _ = torch.min(all_conf, dim=-1)
                # elif self.pts3d_mode == "cam":
                #     depth_conf = output["depth_conf"]
                #     all_conf = torch.stack([depth_conf, sampling_conf], dim=-1)
                #     pts3d_conf, _ = torch.min(all_conf, dim=-1)

                trajectory_conf = repeat(sampling_conf, "b im h w -> b im t h w", t=im) # image per time
                predictions["trajectory_conf"] = trajectory_conf

        if self.enable_gs:
            predictions["3dgs_rendering"] = renderings
            predictions["target_time"] = random_time

        return predictions
    
    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            # del output.ray
            # del output.ray_conf
            output.extrinsics_ray = pred_extrinsic
            output.intrinsics_ray = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            # if "ray" in output:
            #     del output.ray
            # if "ray_conf" in output:
            #     del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output["c2w"] = c2w
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        output["pose_enc"] = pose_enc
        return output


    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        target_gs_traj: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output