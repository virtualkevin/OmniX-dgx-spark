import os
import cv2
import numpy as np
from pathlib import Path
import torch
import moviepy.editor as mpy

from .export_util import export_pointcloud_trajectory

class Visualizer:
    def __init__(
        self,
        save_path: str = "./vis_output",
        save_nums: int = 2,
        export_pc: bool = False,
        export_3dgs: bool = False,
    ):
        self.save_path = save_path
        self.save_nums = save_nums
        self.export_pc = export_pc
        self.export_3dgs = export_3dgs

        os.makedirs(self.save_path, exist_ok=True)
    
    def _create_save_dir(self, epoch, rank, batch_idx, folder_name):
        """Create directory structure: save_path/epoch_{epoch}/rank_{rank}/batch_{batch_idx}/folder_name"""
        save_dir = Path(self.save_path) / f"epoch_{epoch}" / f"rank_{rank}" / f"batch_{batch_idx}" / folder_name
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
        
    def save_images(self, images, epoch, rank, batch_idx, image_folder_name="gt", global_time_idxs=None):
        """images: Tensor [N H W C] or [H W C], global_time_idxs: Tensor [N]"""
        base_dir = Path(self.save_path) / f"epoch_{epoch}" / f"rank_{rank}" / f"batch_{batch_idx}"
        os.makedirs(base_dir, exist_ok=True)
        
        
        if len(images.shape) == 3:
            # Single image [H W C]: save directly without text
            # image_uint8 = (images_np * 255).astype(np.uint8)
            # Convert to numpy if needed
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            else:
                images_np = images.copy()
            image_path = base_dir / f"{image_folder_name}.png"
            
            # Convert RGB to BGR for cv2
            image_bgr = cv2.cvtColor(images_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), image_bgr)
            
        else:
            # Multiple images [N H W C]: add text and save in folder
            save_dir = self._create_save_dir(epoch, rank, batch_idx, image_folder_name)
            # images_with_text = self.add_text_to_image(images, global_time_idxs)
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
            
            for i, image in enumerate(images):
                image_uint8 = (image * 255).astype(np.uint8)
                image_path = save_dir / f"image_{i:02d}.png"
                
                # Convert RGB to BGR for cv2
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)

    def save_videos(self, videos, epoch, rank, batch_idx, video_folder_name="pred_pc_all_image", global_time_idxs=None):
        """videos: Tensor [N T H W C], global_time_idxs: Tensor [N]"""
        save_dir = self._create_save_dir(epoch, rank, batch_idx, video_folder_name)
        
        # Add text and bounds to videos
        videos_with_annotations = self.add_text_and_bound_to_videos(videos, global_time_idxs)
        
        # Convert to numpy if needed
        if isinstance(videos_with_annotations, torch.Tensor):
            videos_with_annotations = videos_with_annotations.cpu().numpy()
        
        N, T, H, W, C = videos_with_annotations.shape
        # for n in range(N):
        #     video_path = save_dir / f"video_{n:02d}.mp4"
            
        #     # OpenCV VideoWriter with very low fps for frame-by-frame viewing
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        #     writer = cv2.VideoWriter(str(video_path), fourcc, fps=2, frameSize=(W, H))
            
        #     for t in range(T):
        #         frame = videos_with_annotations[n, t]
        #         frame_uint8 = (frame * 255).astype(np.uint8)
        #         frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        #         frame_bgr = np.ascontiguousarray(frame_bgr)
        #         writer.write(frame_bgr)
            
        #     writer.release()

        # use moviepy to save videos
        for n in range(N):
            video_path = save_dir / f"video_{n:02d}.mp4"
            
            # 收集当前视频的所有帧
            frames_list = []
            for t in range(T):
                frame = videos_with_annotations[n, t]
   
                if frame.dtype != np.uint8:
                    frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                else:
                    frame_uint8 = frame
                    
                frames_list.append(frame_uint8)

            clip = mpy.ImageSequenceClip(frames_list, fps=2)  # low fps
            clip.write_videofile(
                str(video_path),
                codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p"],  # mac compatible
                audio=False,
                logger=None
            )

    def export_pc_traj(self, trajectory, colors, epoch, rank, batch_idx, pc_folder_name="pred_trajectory", global_time_idxs=None):
        """trajectory: Tensor [N T H W 3], global_time_idxs: Tensor [N]"""
        save_dir = self._create_save_dir(epoch, rank, batch_idx, pc_folder_name)
        
        # Convert to numpy if needed
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()

        export_pointcloud_trajectory(trajectory, colors, save_dir)
    
    def add_text_to_image(self, images, global_time_idxs=None, font_scale=0.6, text_color=(0., 0., 0.), thickness=2):
        """images: Tensor [N H W C], global_time_idxs: Tensor [N]"""
        if isinstance(images, torch.Tensor):
            images = images.clone()
            images_np = images.cpu().numpy()
        else:
            images_np = images.copy()
        
        for i in range(images_np.shape[0]):
            # Convert to uint8 for cv2 operations
            image_uint8 = (images_np[i] * 255).astype(np.uint8)
            image_uint8 = np.ascontiguousarray(image_uint8)

            # Prepare text
            time_idx = global_time_idxs[i].item() if global_time_idxs is not None else 0
            text = f"image_{i:02d}_time_{time_idx:02d}"
            
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text
            cv2.putText(image_uint8, text, (15, 10 + text_height), font, font_scale, text_color, thickness)
            
            # Convert back to float and update
            images_np[i] = image_uint8.astype(np.float32) / 255.0
        
        if isinstance(images, torch.Tensor):
            return torch.from_numpy(images_np).to(images.device)
        return images_np

    def add_text_and_bound_to_videos(self, videos, global_time_idxs=None, font_scale=0.6, text_color=(0.,0.,0.), thickness=2):
        """videos: Tensor [N T H W C], global_time_idxs: Tensor [N], highlight input frame for each image"""
        if isinstance(videos, torch.Tensor):
            videos = videos.clone()
            videos_np = videos.cpu().numpy()
        else:
            videos_np = videos.copy()
        
        N, T, H, W, C = videos_np.shape
        
        for n in range(N):
            target_time = global_time_idxs[n].item() if global_time_idxs is not None else 0
            
            for t in range(T):
                # Convert to uint8 for cv2 operations
                frame = (videos_np[n, t] * 255).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
                
                # Add text
                text = f"image_{n:02d}_time_{t:02d}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # # Draw background rectangle for text
                # cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, text_color, thickness)
                
                # Add red border if this frame corresponds to the target time
                if t == target_time:
                    border_thickness = 8
                    border_color = (255, 0, 0)  # Red border (BGR format)
                    
                    # Draw thick border
                    cv2.rectangle(frame, (0, 0), (W-1, H-1), border_color, border_thickness)
                
                # Convert back to float
                videos_np[n, t] = frame.astype(np.float32) / 255.0
        
        if isinstance(videos, torch.Tensor):
            return torch.from_numpy(videos_np).to(videos.device)
        return videos_np
