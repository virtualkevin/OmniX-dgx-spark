
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import re
import torch
from torch.distributed import all_gather_object
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator
from src.models.model import CustomModel
from src.utils.projection import closed_form_inverse_se3, world_to_camera_coordinates, world_to_pixel_coordinates
from src.eval.cam_pose_metric import calculate_auc
from src.eval.pts3d_metric import accuracy, completion, umeyama
from src.eval.trajectory_metric import compute_tapvid3d_metrics

import open3d as o3d
from PIL import Image
from einops import rearrange
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous

from src.models.utils.scheduler import MultiLinearWarmupCosineAnnealingLR
from src.utils import pylogger
from src.eval.cam_pose_metric import se3_to_relative_pose_error
from src.visualization import Visualizer
from src.visualization.drawing.camera import draw_cameras_with_frustum
from src.visualization.point_cloud import render_all_pointcloud_trajectory, render_per_image_pointcloud_trajectory, draw_per_image_pointcloud_trajectory
from src.visualization.single_channel_map import vis_depth_map

# import wandb
import gc
import time

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class AccumulatedSum(BaseAggregator):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fn="sum",
            default_value=torch.tensor(0.0, dtype=torch.long),
            nan_strategy='warn',
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: int) -> None:
        self.sum_value += value

    def compute(self) -> torch.LongTensor:
        return self.sum_value

class CustomModelWrapper(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        train_criterion: torch.nn.Module,
        validation_criterion: torch.nn.Module,
        optimizer: dict,
        scheduler: dict,
        compile: bool,
        visualizer: Visualizer,
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        enable_depth: bool = False,
        enable_normal: bool = False,
        enable_3dgs: bool = False,
        enable_track: bool = False,
        freeze: Optional[List[str]] = None,
        test_modalities: Optional[List[str]] = None
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'train_criterion', 'validation_criterion'])

        self.net = net
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion

        self.pretrained = pretrained
        self.resume_from_checkpoint = resume_from_checkpoint

        self.enable_depth = enable_depth
        self.enable_normal = enable_normal
        self.enable_3dgs = enable_3dgs
        self.enable_track = enable_track
     
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.freeze = freeze
        
        self.compile = compile
        
        self.visualizer = visualizer
        self.test_modalities = test_modalities
    
        assert self.enable_depth is False, "Depth is not supported yet"
        assert self.enable_normal is False, "Normal is not supported yet"
        assert self.enable_3dgs is False, "3DGS is not supported yet"
        assert self.enable_track is False, "Track is not supported yet"
        # assert self.enable_condition is False, "Condition is not supported yet"

        # use register_buffer to save these with checkpoints
        # so that when we resume training, these bookkeeping variables are preserved
        self.register_buffer("epoch_fraction", torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer("train_total_samples", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("train_total_images", torch.tensor(0, dtype=torch.long, device=self.device))

        self.train_total_samples_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric
        self.train_total_images_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric

        # Initialize camera pose metrics
        self.RRA_thresholds = [5, 15, 30]
        self.RTA_thresholds = [5, 15, 30]

        # pts3d evaluation metrics [pts3d]
        self.pts3d_metrics_per_epoch = {}  # Accumulate all pts3d metrics by dataset and scene for the epoch

        # Trajectory evaluation metrics [trajectory]
        self.trajectory_metrics_per_epoch = {}
        
        # Camera pose metrics [camera_pose]
        self.camera_pose_metrics_per_epoch = {}

        # debug
        self.time_start = 0
    @classmethod
    def load_for_inference(cls, net: CustomModel):
        lit_module = cls(net=net, train_criterion=None, validation_criterion=None, optimizer=None, scheduler=None, compile=False)
        lit_module.eval()
        return lit_module

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        checkpoint["state_dict"] = filtered_state_dict
        print(f"Loaded checkpoint with {len(filtered_state_dict)}/{len(state_dict)} keys")


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""

        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # log gradients, parameter histogram and model topology
                self.wandb_logger.watch(self.net, log="all", log_freq=500, log_graph=False)

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(self.trainer.train_dataloader.dataset, "set_epoch"):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "batch_sampler") and hasattr(self.trainer.train_dataloader.batch_sampler, "set_epoch"):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        # TODO: check the input range of image
        return self.net(batch)
    
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        # if self.global_rank in [0]:
        #     print(f"rank: {self.global_rank}, batch_idx: {batch_idx}")
        #     print(f"Time taken: {time.time() - self.time_start} seconds")
        #     self.time_start = time.time()

        # debug
        if self.global_step % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        B, S, _, H, W = batch["image"].shape
        # forward process
        preds = self.forward(batch)


        # # debug
        # if batch_idx in [58, 59, 60]:
        #     save_root = "/apdcephfs_jn3/share_303535725/yanqinjiang/project/tmp_new"
        #     save_folder = os.path.join(save_root, f"rank_{self.global_rank}_batch_{batch_idx}")
        #     os.makedirs(save_folder, exist_ok=True)
        #     torch.save(batch, os.path.join(save_folder, "batch.pt"))
        #     torch.save(preds, os.path.join(save_folder, "preds.pt"))
            
        #     print(f"rank: {self.global_rank}, batch_idx: {batch_idx}")
        #     print("scene_meta", batch["scene_meta"])
        #     # print(f"loss: {loss}")

        # Compute the loss in higher precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            loss, loss_details = self.train_criterion(batch, preds)
        
        # logging
        if not isinstance(loss, (torch.Tensor, dict, type(None))):  # this will cause a lightning.fabric.utilities.exceptions.MisconfigurationException
            # log loss and the batch information to help debugging
            # use print instead of log because the logger only logs on rank 0, but this could happen on any rank
            print(f"Loss is not a tensor or dict but {type(loss)}, value: {loss}")
            print(f"Loss details: {loss_details}")
            print(f"Batch: {batch}")
            print(f"Batch index: {batch_idx}")
            print(f"Preds: {preds}")
            loss = None  # set loss to None will still break the training loop in DDP, this is intended - we should fix the data to avoid nan loss in the first place
            return loss
        
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.epoch_fraction = torch.tensor(self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches, device=self.device)
        self.log("trainer/epoch", self.epoch_fraction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr_pretrained", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr_new", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[1], on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr_backbone", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[2], on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(f"train_detail_{key}", value, on_step=True, on_epoch=False, prog_bar=False)
                match = re.search(r'/(\d{1,2})$', key)
                if match:
                    stripped_key = key[:match.start()]
                    self.log(f"train/{stripped_key}", value, on_step=True, on_epoch=False, prog_bar=False)
        
        # Log the total number of samples seen so far
        self.train_total_samples_per_step(B)  # aggregate across all GPUs
        self.train_total_samples += self.train_total_samples_per_step.compute()  # accumulate across all steps
        self.train_total_samples_per_step.reset()
        self.log("trainer/total_samples", self.train_total_samples, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of images seen so far
        n_image_cur_step = B * S
        self.train_total_images_per_step(n_image_cur_step)  # aggregate across all GPUs
        self.train_total_images += self.train_total_images_per_step.compute()  # accumulate across all steps
        self.train_total_images_per_step.reset()
        self.log("trainer/total_images", self.train_total_images, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        
        """
        preds: dict
            pts3d: [b im t h w 3]
            pts3d_conf: [b im t h w 3]
            pos_enc: [b im c_enc]
        """

        B, S, _, H, W = batch["image"].shape
        assert B == 1, "Only support batch size 1 for validation"

        # forward
        preds = self.forward(batch)
        
        # evaluate 3D points
        if "pts3d" in self.test_modalities:
            self.evaluate_pts3d(batch, preds)
        
        # evaluate relative camera poses
        if "camera_pose" in self.test_modalities:
            self.evaluate_camera_poses(batch, preds)

        # evaluate trajectory APD_3D
        if "trajectory" in self.test_modalities:
            self.evaluate_trajectory(batch, preds)

        # visualize pc_traj and export
        if self.visualizer is not None:
            if self.trainer.sanity_checking:
                batch_idx = batch_idx - 1000
            self.visualize_dynamic_mask(batch, preds, batch_idx)
            self.visualize_trajectory(batch, preds, batch_idx)
            self.visualize_camera_poses(batch, preds, batch_idx)

        # clean
        del preds
        torch.cuda.empty_cache()

        return
    
    def evaluate_pts3d(self, batch, preds):

        dataset_name = batch["scene_meta"]["dataset_name"][0]
        data_sampling_type = batch["scene_meta"]["data_sampling_type"][0]
        scene_path = batch["scene_meta"]["scene_path"][0]
        
        colors = batch["image"].permute(0, 1, 3, 4, 2).cpu().numpy()
        image_idxs = batch["image_info"][0][:,1]
        global_time_idxs = batch["image_info"][0][:, 3]

        # trajectory [b im t h w 3] -> pts3d[b im h w 3]
        # to float
        preds["trajectory"] = preds["trajectory"].float()
        pred_pts3d = preds["trajectory"][:, image_idxs,global_time_idxs].cpu().numpy() 
        gt_pts3d = batch["trajectory"][:, image_idxs, global_time_idxs].cpu().numpy()
        valid_mask = batch["valid_mask"]
        valid_mask = valid_mask.cpu().numpy()
        valid_mask = valid_mask & (~(np.isnan(gt_pts3d).any(axis=-1)))

        assert pred_pts3d.shape == gt_pts3d.shape, f"Predicted points shape {pred_pts3d.shape} does not match ground truth shape {gt_pts3d.shape}." 

        # filter invalid point
        colors = colors[valid_mask] # [n 3]
        pred_pts3d = pred_pts3d[valid_mask]
        gt_pts3d = gt_pts3d[valid_mask]        

        # coarse align
        c, R, t = umeyama(pred_pts3d.T, gt_pts3d.T)
        pred_pts3d = c * np.einsum('nj, ij -> ni', pred_pts3d, R) + t.T
        
        # get pred and gt point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_pts3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts3d)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)

        # ICP alignment
        if "dtu" in dataset_name:
            threshold = 100
        else:
            threshold = 0.1
        
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_gt,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        transformation = reg_p2p.transformation
        pcd = pcd.transform(transformation)

        pcd.estimate_normals()
        pcd_gt.estimate_normals()
        pred_normal = np.asarray(pcd.normals)
        gt_normal = np.asarray(pcd_gt.normals)
        
        # compute metrics
        pred_points_np = np.asarray(pcd.points)
        gt_points_np = np.asarray(pcd_gt.points)

        acc, acc_med, nc1, nc1_med = accuracy(
            gt_points_np, pred_points_np, gt_normal, pred_normal
        )
        comp, comp_med, nc2, nc2_med = completion(
            gt_points_np, pred_points_np, gt_normal, pred_normal
        )
        print(f"Accuracy: {acc:.4f}, Accuracy median: {acc_med:.4f}. scene_name: {scene_path}")
        print(f"Completion: {comp:.4f}, Completion median: {comp_med:.4f}. scene_name: {scene_path}")
        print(f"Normal consistency 1: {nc1:.4f}, Normal consistency 1 median: {nc1_med:.4f}. scene_name: {scene_path}")
        print(f"Normal consistency 2: {nc2:.4f}, Normal consistency 2 median: {nc2_med:.4f}. scene_name: {scene_path}")
        
        result = {
            scene_path: {
                "accuracy": acc, "accuracy_median": acc_med,
                "completion": comp, "completion_median": comp_med,
                "nc1": nc1, "nc1_median": nc1_med,
                "nc2": nc2, "nc2_median": nc2_med,
            }
        }
        # Collect metrics for the scene
        dataset_tag = f"{dataset_name}"
        if dataset_tag not in self.pts3d_metrics_per_epoch:
            self.pts3d_metrics_per_epoch[dataset_tag] = {}

        self.pts3d_metrics_per_epoch[dataset_tag].update(result)

        return
    
    def evaluate_camera_poses(self, batch, preds):
        dataset_name = batch["scene_meta"]["dataset_name"][0]
        data_sampling_type = batch["scene_meta"]["data_sampling_type"][0]
        scene_path = batch["scene_meta"]["scene_path"][0]

        gt_cameras = batch["camera_pose"]
        gt_cameras = closed_form_inverse_se3(gt_cameras) # w2c
        h, w = batch["image"].shape[-2:]
        pred_cameras, _  = pose_encoding_to_extri_intri(preds["pose_enc"], (h, w))
        pred_cameras = affine_inverse(pred_cameras)
        pred_cameras = as_homogeneous(pred_cameras)

        # compute the metrics: RRA, RTA, mAA
        # Ensure we have enough poses to compute relative errors
        if pred_cameras.shape[1] >= 2:
            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_cameras[0], gt_cameras[0], len(pred_cameras[0]))
            result = {}
            result[scene_path]= {}
            for tau in self.RRA_thresholds:
                result[scene_path][f"RRA_at_{tau}"] = (rel_rangle_deg < tau).float().mean().item()
            for tau in self.RTA_thresholds:
                result[scene_path][f"RTA_at_{tau}"] = (rel_tangle_deg < tau).float().mean().item()
            # Compute mAA(30)
            result[scene_path]['mAA_30'] = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30).item()

        else:
            log.warning("Not enough camera poses to compute relative errors.")

        # Collect metrics for the scene
        dataset_tag = f"{dataset_name}"
        if dataset_tag not in self.camera_pose_metrics_per_epoch:
            self.camera_pose_metrics_per_epoch[dataset_tag] = {}
        
        self.camera_pose_metrics_per_epoch[dataset_tag].update(result)

        return
        
    def evaluate_trajectory(self, batch, preds):
        dataset_name = batch["scene_meta"]["dataset_name"][0]
        data_sampling_type = batch["scene_meta"]["data_sampling_type"][0]
        scene_path = batch["scene_meta"]["scene_path"][0]

        # Convert trajectory to camera coordinates
        b, im, t, h, w, _ = preds["trajectory"].shape
        pred_camera_pose, _  = pose_encoding_to_extri_intri(preds["pose_enc"], (h, w))
        # pose_3x4 to pose_4x4 in inverse_se3
        pred_camera_pose = as_homogeneous(pred_camera_pose)
        pred_trajectory = world_to_camera_coordinates(preds["trajectory"][0], pred_camera_pose[0], has_batch=True)
     
        gt_trajectory = world_to_camera_coordinates(batch["trajectory"][0], batch["camera_pose"][0],has_batch=True)
        pred_trajectory = pred_trajectory.detach().cpu().numpy()
        gt_trajectory = gt_trajectory.detach().cpu().numpy()

        valid_mask = batch["valid_mask"].cpu().numpy() # [b im h w]
        trajectory_foreground = batch["trajectory_foreground"].cpu().numpy()
        # only eval foreground
        tmp_idx = np.arange(im)
        foreground_mask = trajectory_foreground[:, tmp_idx, tmp_idx]
        valid_mask = valid_mask & foreground_mask

        valid_mask = valid_mask[:, :, None] & (~np.isnan(gt_trajectory).any(axis=-1))
        gt_occluded = ~valid_mask
        pred_occluded = ~valid_mask

        pred_trajectory = pred_trajectory.reshape(im, t, h*w, 3)
        gt_trajectory = gt_trajectory.reshape(im, t, h*w, 3)

        gt_occluded = gt_occluded.reshape(b*im, t, h*w)
        pred_occluded = pred_occluded.reshape(b*im, t, h*w)

        intrinsics = batch["intrinsic"].cpu().numpy() # [b im 3 3]
        intrinsics = intrinsics.reshape(b*im, 3, 3)
        intrinsics_params = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1], \
            intrinsics[:, 0, 2], intrinsics[:, 1, 2]], axis=-1)

        # Compute metrics
        # we use gt_intrinsic_params, since pred_trajectory will be scaled to gt_trajectory
        eval_result = compute_tapvid3d_metrics(gt_occluded, gt_trajectory, pred_occluded, pred_trajectory, intrinsics_params, order='b t n')

        # we only need APD_3D, average by im
        result = {}
        result[scene_path] = {}
        for metric_name, metirc_val in eval_result.items():
            if "pts_within_" in metric_name:
                result[scene_path][metric_name] = np.mean(metirc_val)

        dataset_tag = f"{dataset_name}"
        if dataset_tag not in self.trajectory_metrics_per_epoch:
            self.trajectory_metrics_per_epoch[dataset_tag] = {}

        self.trajectory_metrics_per_epoch[dataset_tag].update(result)

    def on_validation_epoch_end(self) -> None:

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction, sync_dist=True)
        self.log("trainer/total_samples", self.train_total_samples.cpu().item(), sync_dist=True)
        self.log("trainer/total_images", self.train_total_images.cpu().item(), sync_dist=True)

        # Log the 3D reconstruction metrics
        self.aggregate_and_log_pts3d_metrics()
        # Log the camera pose metrics
        self.aggregate_and_log_camera_metrics()
        # Log the trajectory metrics
        self.aggregate_and_log_trajectory_metrics()
    

    def aggregate_and_log_pts3d_metrics(self,):
        # Gather and deduplicate metrics by dataset across all ranks after all batches
        if torch.distributed.is_initialized():
            self.pts3d_metrics_per_epoch = gather_deduplicated_scene_metrics(self.pts3d_metrics_per_epoch)
        
        for dataset_tag, scenes in self.pts3d_metrics_per_epoch.items():
            acc_list = [metrics["accuracy"] for metrics in scenes.values()]
            acc_med_list = [metrics["accuracy_median"] for metrics in scenes.values()]
            comp_list = [metrics["completion"] for metrics in scenes.values()]
            comp_med_list = [metrics["completion_median"] for metrics in scenes.values()]
            nc1_list = [metrics["nc1"] for metrics in scenes.values()]
            nc1_med_list = [metrics["nc1_median"] for metrics in scenes.values()]
            nc2_list = [metrics["nc2"] for metrics in scenes.values()]
            nc2_med_list = [metrics["nc2_median"] for metrics in scenes.values()]

            # Log global aggregated metrics per dataset
            mean_accuracy = np.mean(acc_list)
            median_accuracy = np.mean(acc_med_list)
            mean_completion = np.mean(comp_list)
            median_completion = np.mean(comp_med_list)
            mean_nc1 = np.mean(nc1_list)
            median_nc1 = np.mean(nc1_med_list)
            mean_nc2 = np.mean(nc2_list)
            median_nc2 = np.mean(nc2_med_list)

            self.log(f"val_pts3d_{dataset_tag}/accuracy", mean_accuracy, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/accuracy_median", median_accuracy, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/completion", mean_completion, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/completion_median", median_completion, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/nc1", mean_nc1, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/nc1_median", median_nc1, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/nc2", mean_nc2, sync_dist=True)
            self.log(f"val_pts3d_{dataset_tag}/nc2_median", median_nc2, sync_dist=True)

        # Clear all dataset metrics after logging
        self.pts3d_metrics_per_epoch.clear()

        return
    
    def aggregate_and_log_camera_metrics(self):
        # Gather and deduplicate metrics by dataset across all ranks after all batches
        if torch.distributed.is_initialized():
            self.camera_pose_metrics_per_epoch = gather_deduplicated_scene_metrics(self.camera_pose_metrics_per_epoch)

        # Aggregate global metrics per dataset using deduplicated data
        for dataset_tag, scenes in self.camera_pose_metrics_per_epoch.items():
            RRA_5 = [value[f"RRA_at_{self.RRA_thresholds[0]}"] for value in scenes.values()]
            RRA_15 = [value[f"RRA_at_{self.RRA_thresholds[1]}"] for value in scenes.values()]
            RRA_30 = [value[f"RRA_at_{self.RRA_thresholds[2]}"] for value in scenes.values()]
            RTA_5 = [value[f"RTA_at_{self.RTA_thresholds[0]}"] for value in scenes.values()]
            RTA_15 = [value[f"RTA_at_{self.RTA_thresholds[1]}"] for value in scenes.values()]
            RTA_30 = [value[f"RTA_at_{self.RTA_thresholds[2]}"] for value in scenes.values()]
            mAA_30 = [value["mAA_30"] for value in scenes.values()]

            # Log global aggregated metrics per dataset
            RRA_5 = np.mean(RRA_5)
            RRA_15 = np.mean(RRA_15)
            RRA_30 = np.mean(RRA_30)
            RTA_5 = np.mean(RTA_5)
            RTA_15 = np.mean(RTA_15)
            RTA_30 = np.mean(RTA_30)
            mAA_30 = np.mean(mAA_30)

            self.log(f"val_camera_pose_{dataset_tag}/RRA_5", RRA_5, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/RRA_15", RRA_15, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/RRA_30", RRA_30, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/RTA_5", RTA_5, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/RTA_15", RTA_15, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/RTA_30", RTA_30, sync_dist=True)
            self.log(f"val_camera_pose_{dataset_tag}/mAA_30", mAA_30, sync_dist=True)

        # Clear all dataset metrics after logging
        self.camera_pose_metrics_per_epoch.clear()
    
    def aggregate_and_log_trajectory_metrics(self):
        # Gather and deduplicate metrics by dataset across all ranks after all batches
        if torch.distributed.is_initialized():
            self.trajectory_metrics_per_epoch = gather_deduplicated_scene_metrics(self.trajectory_metrics_per_epoch)

        # Aggregate global metrics per dataset using deduplicated data
        for dataset_tag, scenes in self.trajectory_metrics_per_epoch.items():
            APD3D_1 = [value[f"pts_within_1"] for value in scenes.values()]
            APD3D_2 = [value[f"pts_within_2"] for value in scenes.values()]
            APD3D_4 = [value[f"pts_within_4"] for value in scenes.values()]
            APD3D_8 = [value[f"pts_within_8"] for value in scenes.values()]
            APD3D_16 = [value[f"pts_within_16"] for value in scenes.values()]
            APD3D = [value[f"average_pts_within_thresh"] for value in scenes.values()]

            # Log global aggregated metri`cs per dataset
            APD3D_1 = np.mean(APD3D_1)
            APD3D_2 = np.mean(APD3D_2)
            APD3D_4 = np.mean(APD3D_4)
            APD3D_8 = np.mean(APD3D_8)
            APD3D_16 = np.mean(APD3D_16)
            APD3D = np.mean(APD3D)

            self.log(f"val_trajectory_{dataset_tag}/APD3D_1", APD3D_1, sync_dist=True)
            self.log(f"val_trajectory_{dataset_tag}/APD3D_2", APD3D_2, sync_dist=True)
            self.log(f"val_trajectory_{dataset_tag}/APD3D_4", APD3D_4, sync_dist=True)
            self.log(f"val_trajectory_{dataset_tag}/APD3D_8", APD3D_8, sync_dist=True)
            self.log(f"val_trajectory_{dataset_tag}/APD3D_16", APD3D_16, sync_dist=True)
            self.log(f"val_trajectory_{dataset_tag}/APD3D", APD3D, sync_dist=True)

        # Clear all dataset metrics after logging
        self.trajectory_metrics_per_epoch.clear()

    def configure_optimizers(self):
        # freeze model
        if self.freeze is not None and len(self.freeze) > 0:
            for name, param in self.net.named_parameters():
                for freeze_mudule in self.freeze:
                    if freeze_mudule in name:
                        param.requires_grad_(False)
                        log.info(f"freeze {name}module weights!")
        pretrained_params = []
        backbone_params = []
        new_params = []
        # TODO: check this, we will add new adapter
        pretrained_params_list =  self.hparams.optimizer.pretrained_params_list
        new_params_list = self.hparams.optimizer.new_params_list
        
        for name, param in self.net.named_parameters():
            
            if not param.requires_grad:
                continue
            if any(pretrained_param in name for pretrained_param in pretrained_params_list):
                pretrained_params.append(param)
            elif any(new_param in name for new_param in new_params_list):
                new_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer_config = self.hparams.optimizer
        pretrained_lr = float(optimizer_config.param_groups.pretrained.lr)
        pretrained_wd = float(optimizer_config.param_groups.pretrained.weight_decay)
        backbone_lr = float(optimizer_config.param_groups.backbone.lr)
        backbone_wd = float(optimizer_config.param_groups.backbone.weight_decay)
        new_lr = float(optimizer_config.param_groups.new.lr)
        new_wd = float(optimizer_config.param_groups.new.weight_decay)
        
        param_groups = [
            {
                "params": pretrained_params,
                "lr": pretrained_lr,
                "weight_decay": pretrained_wd
            },
            {
                "params": new_params,
                "lr": new_lr,
                "weight_decay": new_wd
            },
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": backbone_wd
            }

        ]
        optimizer_args = {}
        if hasattr(optimizer_config, "betas"):
            optimizer_args["betas"] = tuple(float(b) for b in optimizer_config.betas)
        
        optimizer = torch.optim.AdamW(param_groups, **optimizer_args)
        
        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler
            original_warmup_epochs = scheduler_config.warmup_epochs
            original_max_epochs = scheduler_config.max_epochs
            eta_min_ratio = scheduler_config.eta_min_ratio
            total_steps = self.trainer.estimated_stepping_batches
            scaled_warmup_steps = int(original_warmup_epochs * total_steps / original_max_epochs)
            scaled_max_steps = total_steps
            scheduler_kwargs = ({
                'warmup_steps': scaled_warmup_steps,
                'max_steps': scaled_max_steps,
                'eta_min_ratio': eta_min_ratio
            })
            
            scheduler = MultiLinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                **scheduler_kwargs
            )

            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "train/lr"
                    }
                }
        else:
            return {
                    "optimizer": optimizer
                }

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        # Load pretrained weights if available and not resuming
        # note that if resume_from_checkpoint is set, the Trainer is responsible for actually loading the checkpoint
        # so we are only using resume_from_checkpoint as a check of whether we should load the pretrained weights
        if self.pretrained and not self.resume_from_checkpoint:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        log.info(f"Loading pretrained: {self.pretrained}")
        if isinstance(self.net, CustomModel):
            log.info(f"Loading pretrained weights from {self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location=torch.device('cpu'), weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # load checkpoint
            if sum(key.startswith("net.") for key in state_dict) / len(state_dict) > 0.9:
                filtered_state_dict = {key[4:]:val for key, val in state_dict.items() if key.startswith("net.")}
            else:
                # load checkpoint
                filtered_keys = []
        
                if not self.net.enable_gs:
                    filtered_keys.append("gs_head")
                    # filtered_keys.append("gs_offset_head")
                if not hasattr(self.net, "cam_enc"):
                    filtered_keys.append("cam_enc")
            
                filtered_state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in filtered_keys)}

                # add new heads, offset_head using feature_only
                new_state_dict = {k.replace("head", "offset_head"):v for k, v in filtered_state_dict.items() if ("head" in k) and ("scratch.output_conv2.2" not in k) and ("aux" not in k)}
                filtered_state_dict.update(new_state_dict)
          
            # Load the filtered state_dict into the new model
            # missing keys contains: aux.1.2 / 2.2 /3.2 is expected, because they share ln
            missing_keys, unexpected_keys = self.net.load_state_dict(filtered_state_dict, strict=False)
            log.info(f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
    
    def visualize_trajectory(self, batch, preds, batch_idx=0):
        """Visualize 4D point cloud sequence"""

        trajectory = preds['trajectory'][0] # [im t h w xyz]
        images = batch["image"][0] # [im c h w]
        images = rearrange(images, 'im c h w -> im h w c')
        global_time_idxs = batch["image_info"][0][:, 3]

        # render all_image point cloud
        pred_pc_traj_images, pred_pc_traj_depths = render_all_pointcloud_trajectory(
            trajectory,
            batch["intrinsic"][0],
            batch["camera_pose"][0],
            images,
            return_depth=True
        )
        vis_pred_pc_traj_depths = vis_depth_map(pred_pc_traj_depths)

        self.visualizer.save_videos(pred_pc_traj_images, self.current_epoch, self.global_rank, 
                    batch_idx=batch_idx, video_folder_name="pred_pc_all_image", global_time_idxs=global_time_idxs)
        self.visualizer.save_videos(vis_pred_pc_traj_depths, self.current_epoch, self.global_rank, 
                    batch_idx=batch_idx, video_folder_name="pred_pc_all_image_depth", global_time_idxs=global_time_idxs)
        
        # render trajectory
        # first we should render per_image point cloud
        pred_pc_traj_images_per_image = render_per_image_pointcloud_trajectory(
            trajectory,
            batch["intrinsic"][0],
            batch["camera_pose"][0],
            images,
            return_depth=False
        )

        # then we sample trajectory and render it
        pred_pc_traj_images_per_image_with_traj = draw_per_image_pointcloud_trajectory(
            trajectory,
            batch["intrinsic"][0],
            batch["camera_pose"][0],
            render_images=pred_pc_traj_images_per_image,
        )
        self.visualizer.save_videos(pred_pc_traj_images_per_image_with_traj, self.current_epoch, self.global_rank, 
                    batch_idx=batch_idx, video_folder_name="pred_pc_per_image_with_traj", global_time_idxs=global_time_idxs)
        # logging
        # we should log gt images as well
        self.visualizer.save_images(images, self.current_epoch, self.global_rank, 
                    batch_idx=batch_idx, image_folder_name="gt_images", global_time_idxs=global_time_idxs)

        # export trajectory if need
        # if batch_idx < self.visualizer.save_nums:
        #     self.visualizer.export_pc_traj(trajectory, images, self.current_epoch, self.global_rank, batch_idx, pc_folder_name="pred_trajectory")

    def visualize_camera_poses(self, batch, preds, batch_idx=0):
     
        """Visualize camera poses"""
        gt_camera_poses = batch["camera_pose"][0]
        h, w = batch["image"].shape[-2:]
        pred_camera_poses, _ = pose_encoding_to_extri_intri(preds["pose_enc"], (h, w))
        # pose_3x4 to pose_4x4 in inverse_se3
        # pred_camera_poses = closed_form_inverse_se3(pred_camera_extrinsics)
        pred_camera_poses = as_homogeneous(pred_camera_poses)
        pred_camera_poses = pred_camera_poses[0]
        
        gt_color = torch.zeros(3, dtype=torch.float32)
        pred_color = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        gt_colors = torch.stack([gt_color] * len(gt_camera_poses))
        pred_colors = torch.stack([pred_color] * len(pred_camera_poses))

        camera_poses_all = torch.cat([gt_camera_poses, pred_camera_poses], dim=0)
        colors_all = torch.cat([gt_colors, pred_colors], dim=0)

        camera_visualization = draw_cameras_with_frustum(
            camera_poses_all, colors_all,
            line_width=3, point_radius=3, font_size=24
        ) # [h w*3 3]

        # logging
        self.visualizer.save_images(camera_visualization, self.current_epoch, self.global_rank, 
                    batch_idx=batch_idx, image_folder_name="camera_poses")
        
    
    def visualize_dynamic_mask(self, batch, preds, batch_idx=0):
        _, im, _, h, w = batch["image"].shape
        patch_size = self.net.offset_head.patch_size
        resize_factor = self.net.offset_head.resize_factor
        hp = h // patch_size
        wp = w // patch_size
        global_time_idxs = batch["image_info"][0][:, 3]
        dynamic_token_masks = preds["dynamic_token_mask"][0].reshape(im, hp*resize_factor, wp*resize_factor)
        dynamic_token_masks = dynamic_token_masks.unsqueeze(3).repeat(1, 1, 1, 3).float()
        self.visualizer.save_images(dynamic_token_masks, self.current_epoch, self.global_rank, 
            batch_idx=batch_idx, image_folder_name="dynamic_token_masks", global_time_idxs=global_time_idxs)
        
        pts3d_dynamic_score = preds["pts3d_dynamic_score"][0]
        pts3d_dynamic_score = pts3d_dynamic_score.unsqueeze(3).repeat(1, 1, 1, 3).float()
        self.visualizer.save_images(pts3d_dynamic_score, self.current_epoch, self.global_rank, 
            batch_idx=batch_idx, image_folder_name="pts3d_dynamic_score", global_time_idxs=global_time_idxs)
    
    # def on_after_backward(self):
    #     # 遍历所有参数
    #     for name, param in self.named_parameters():
    #         # 1. 只有那些需要梯度的参数才可能是"unused"
    #         # 2. 如果 param.grad 是 None，说明它没参与 Loss 计算
    #         if param.requires_grad and param.grad is None:
    #             print(f"⚠️ 发现未使用的参数 (Unused Parameter): {name}")
    #     import ipdb
    #     ipdb.set_trace()
    
def gather_deduplicated_scene_metrics(reconstruction_metrics_per_epoch):
    """Gathers and deduplicates scene-specific metrics across all ranks by dataset."""
    gathered_metrics = [None] * torch.distributed.get_world_size()
    all_gather_object(gathered_metrics, reconstruction_metrics_per_epoch)

    # Flatten and deduplicate metrics across all ranks
    all_metrics = {}
    for rank_metrics in gathered_metrics:
        for dataset_name, scenes in rank_metrics.items():
            if dataset_name not in all_metrics:
                all_metrics[dataset_name] = {}
            all_metrics[dataset_name].update(scenes)  # Keeps the first occurrence of each scene

    return all_metrics