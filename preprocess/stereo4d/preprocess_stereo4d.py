import argparse
import math
import os
import os.path as osp
import glob
from functools import partial

from absl import logging
import cv2
import jax
from jax import numpy as jnp
import jaxcam
import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map

# 强制 JAX 使用 CPU
jax.config.update('jax_platform_name', 'cpu')

# --- 1. 数据加载函数 (已更新) ---
def load_dataset_npz(path):
  with open(path, 'rb') as f:
    data_zip = np.load(f)
    data = {}
    for k in data_zip.keys():
      data[k] = data_zip[k]
  
  # Intrinsics / FOV
  data['meta_fov'] = {
    'start_yaw_in_degrees': data['fov_bounds'][0],
    'end_yaw_in_degrees': data['fov_bounds'][1],
    'start_tilt_in_degrees': data['fov_bounds'][2],
    'end_tilt_in_degrees': data['fov_bounds'][3],
  }
  data.pop('fov_bounds')

  # Poses
  c2w = data['camera2world']  # (T, 3, 4)
  R = c2w[:, :, :3]
  t = c2w[:, :, 3:]
  
  # Compute W2C
  R_inv = np.transpose(R, (0, 2, 1))
  t_inv = -np.matmul(R_inv, t)
  data['extrs_rectified'] = np.concatenate([R_inv, t_inv], axis=-1)
  
  # Keep c2w!
  # data.pop('camera2world') 

  # Tracks
  if 'track_lengths' in data:
      lengths = data['track_lengths']
      shape = (len(lengths), len(data['timestamps']), 3)
      tracks = np.full(shape, np.nan, dtype=np.float32)
      tracks[
        np.repeat(np.arange(lengths.shape[0]), lengths),
        data['track_indices'], :
      ] = data['track_coordinates']
      data['track3d'] = tracks # [N, T, 3]
      
      data.pop('track_lengths')
      data.pop('track_indices')
      data.pop('track_coordinates')
  else:
      data['track3d'] = np.zeros((0, len(data['timestamps']), 3))

  return data
class EquiVideoLoader:
  def __init__(self, video_id, raw_video_folder):
    self.video_path = osp.join(raw_video_folder, video_id + '.mp4')
    # 设定目标尺寸 (宽, 高)
    self.target_size = (4096, 2048)

  def retrieve_frames_cv2(self, timestamps):
    vidcap = cv2.VideoCapture(self.video_path)
    video = []
    
    if vidcap.isOpened(): 
      width  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   
      height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
      # 检查是否需要 resize，减少不必要的计算
      needs_resize = (int(width) != self.target_size[0] or int(height) != self.target_size[1])
      if needs_resize:
          logging.info(f"Video resolution {int(width)}x{int(height)} will be resized to {self.target_size[0]}x{self.target_size[1]}")
    else:
        raise Exception(f'vidcap error: video is not opened at {self.video_path}')
    
    for timestamp in tqdm.tqdm(timestamps, desc='Extract frames'):
      vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp / 1000)
      success, image = vidcap.read()
      if not success: 
        print(f"Warning: Failed at {timestamp}")
        continue 
      
      # 如果尺寸不符，进行 resize
      if needs_resize:
        # cv2.resize 的参数顺序是 (width, height)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
      video.append(image[..., ::-1]) # BGR to RGB
      
    vidcap.release() # 读取完毕释放资源
    return np.stack(video, axis=0)

# --- 2. 几何计算函数 ---

def get_perspective_intrinsics(hfov_degrees, height, width):
    focal_length = width * 0.5 / np.tan(0.5 * np.deg2rad(hfov_degrees))
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def get_equirect_rectification_map(equirect_hw, meta_fov, rectified2rig):
  longitude_stereo = np.linspace(
      np.radians(meta_fov['start_yaw_in_degrees']),
      np.radians(meta_fov['end_yaw_in_degrees']),
      equirect_hw[1],
  )
  latitude_stereo = np.linspace(
      np.radians(meta_fov['start_tilt_in_degrees']),
      np.radians(meta_fov['end_tilt_in_degrees']),
      equirect_hw[0],
  )
  xv, yv = np.meshgrid(longitude_stereo, latitude_stereo)
  ray_x = np.cos(yv) * np.sin(xv)
  ray_y = np.sin(yv)
  ray_z = np.cos(yv) * np.cos(xv)

  ray_stereo = np.stack([ray_x, ray_y, ray_z], axis=0)
  ray_rig = np.einsum('ij,jhw->ihw', rectified2rig, ray_stereo)

  lon = np.arctan2(ray_rig[0], ray_rig[2])
  lat = np.arcsin(ray_rig[1])

  u = ((lon - np.radians(meta_fov['start_yaw_in_degrees'])) / 
       np.radians(meta_fov['end_yaw_in_degrees'] - meta_fov['start_yaw_in_degrees']) * (equirect_hw[1] - 1))
  v = ((lat - np.radians(meta_fov['start_tilt_in_degrees'])) / 
       np.radians(meta_fov['end_tilt_in_degrees'] - meta_fov['start_tilt_in_degrees']) * (equirect_hw[0] - 1))

  return u.astype(np.float32), v.astype(np.float32)

def rectify_equirect_frame_wrapper(kwargs):
  image = kwargs['image']
  meta_fov = kwargs['meta_fov']
  corrections = kwargs['corrections']
  
  left = image[:, : image.shape[1] // 2]
  right = image[:, image.shape[1] // 2 :]
  
  xx, yy = get_equirect_rectification_map(left.shape[:2], meta_fov, corrections['rectified2rig_left'])
  rect_left = cv2.remap(left, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  
  xx, yy = get_equirect_rectification_map(right.shape[:2], meta_fov, corrections['rectified2rig_right'])
  rect_right = cv2.remap(right, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  
  return np.concatenate([rect_left, rect_right], axis=1)

class CropFlags:
  output_hfov = 120
  imh = 512
  imw = 512
  meta_fov = None

def field_of_view_to_focal_length(fov_degrees, size):
  return size * 0.5 / np.tan(0.5 * (math.pi / 180.0) * fov_degrees)

def create_jaxcam(hfov, height, width):
  fx = field_of_view_to_focal_length(hfov, width)
  # 简化的 Rotation Matrix (Identity for Yaw/Pitch/Roll=0)
  # r_x=I, r_y=I, r_z=I -> R=I.T -> I
  orientation = jnp.eye(3) 
  
  camera = jaxcam.Camera.create(
      orientation=orientation,
      position=jnp.zeros(3),
      focal_length=jnp.asarray(fx),
      principal_point=jnp.asarray([0.5 * width, 0.5 * height]),
      image_size=(jnp.asarray([width, height], dtype=jnp.float32)),
      pixel_aspect_ratio=1.0,
      radial_distortion=None,
      is_fisheye=False,
  )
  return camera

def equirectangular_to_jaxcam_map(equirect_hw, meta_fov, camera):
  width, height = camera.image_size.astype(int)
  x = np.linspace(0, width - 1, width) + 0.5
  y = np.linspace(0, height - 1, height) + 0.5
  xv, yv = np.meshgrid(x, y)
  rays = jaxcam.pixels_to_rays(camera, np.stack([xv, yv], axis=-1), normalize=True)
  
  lon = np.arctan2(rays[..., 0], rays[..., 2])
  lat = np.arcsin(rays[..., 1])

  u = ((lon - np.radians(meta_fov['start_yaw_in_degrees'])) / np.radians(meta_fov['end_yaw_in_degrees'] - meta_fov['start_yaw_in_degrees']) * (equirect_hw[1] - 1))
  v = ((lat - np.radians(meta_fov['start_tilt_in_degrees'])) / np.radians(meta_fov['end_tilt_in_degrees'] - meta_fov['start_tilt_in_degrees']) * (equirect_hw[0] - 1))
  return u.astype(np.float32), v.astype(np.float32)

def crop_perspective_wrapper(kwargs):
  img_lr = kwargs['rectified_equirect_left_right']
  cf = kwargs['crop_flag']
  
  # Crop Left Only
  left = img_lr[:, : img_lr.shape[1] // 2]
  camera = create_jaxcam(cf.output_hfov, cf.imh, cf.imw)
  xx, yy = equirectangular_to_jaxcam_map(left.shape[:2], cf.meta_fov, camera)
  
  return cv2.remap(left, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# --- 3. 主流程 ---

def process_video(vid: str, output_dir: str, raw_video_folder: str, npz_folder: str, output_hfov: float):
  # 1. 路径设置
  base_out_dir = osp.join(output_dir, vid, "cam_left")
  os.makedirs(osp.join(base_out_dir, "image"), exist_ok=True)
  os.makedirs(osp.join(base_out_dir, "camera"), exist_ok=True)
  os.makedirs(osp.join(base_out_dir, "trajectory"), exist_ok=True)
  # 2. 加载数据
  npz_path = osp.join(npz_folder, f'{vid}.npz')
  logging.info(f"Loading {npz_path}...")
  dp = load_dataset_npz(npz_path) # 使用内部定义的函数
  
  meta_fov = dp['meta_fov']
  corrections = {
    'rectified2rig_left': dp['rectified2rig'][0],
    'rectified2rig_right': dp['rectified2rig'][1],
  }
  
  # 3. 准备 Camera Pose [T, 4, 4]
  c2w_raw = dp['camera2world'] # [T, 3, 4] (现在已保留)
  T = c2w_raw.shape[0]
  bottom_rows = np.repeat(np.array([0,0,0,1], dtype=np.float32).reshape(1,1,4), T, axis=0)
  camera_poses_all = np.concatenate([c2w_raw, bottom_rows], axis=1)

  # 4. 准备 Trajectory [T, N, 3]
  # Loader 返回的是 [N, T, 3]，我们需要转置
  if 'track3d' in dp and dp['track3d'].size > 0:
      trajectory_all = dp['track3d'].transpose(1, 0, 2) # -> [T, N, 3]
  else:
      trajectory_all = np.zeros((T, 0, 3))
      logging.warning("No track3d data found or empty.")

  # 5. 准备内参
  im_h, im_w = 512, 512
  intrinsic_matrix = get_perspective_intrinsics(output_hfov, im_h, im_w)

  # 6. 加载视频并 Rectify
  timestamps = dp['timestamps']
  raw_video_id = str(dp['video_id'])
  equi_loader = EquiVideoLoader(raw_video_id, raw_video_folder)
  equi_video = equi_loader.retrieve_frames_cv2(timestamps)

  logging.info('Rectifying...')
  rectified_video = process_map(
    rectify_equirect_frame_wrapper,
    [{'image': x, 'meta_fov': meta_fov, 'corrections': corrections} for x in equi_video],
    max_workers=4, chunksize=2, desc='Rectify'
  )
  rectified_video = np.stack(rectified_video, axis=0)

  # 7. Crop Perspective
  logging.info('Cropping...')
  crop_flag = CropFlags()
  crop_flag.meta_fov = meta_fov
  crop_flag.output_hfov = output_hfov
  crop_flag.imh = im_h
  crop_flag.imw = im_w
  
  left_pers_images = process_map(
    crop_perspective_wrapper,
    [{'rectified_equirect_left_right': x, 'crop_flag': crop_flag} for x in rectified_video],
    max_workers=8, chunksize=2, desc='Crop'
  )

  # 8. 保存
  logging.info(f'Saving to {base_out_dir}...')
  for idx, img in enumerate(tqdm.tqdm(left_pers_images, desc="Saving")):
      # 保存图像
      cv2.imwrite(
          osp.join(base_out_dir, "image", f"frame_{idx:04d}.png"), 
          cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      )
      
      # 保存 NPZ
      pose = camera_poses_all[idx] if idx < T else np.eye(4)
      np.savez(
          osp.join(base_out_dir, "camera", f"frame_{idx:04d}.npz"),
          intrinsic=intrinsic_matrix,
          camera_pose=pose,
      )

      np.savez(
         osp.join(base_out_dir, "trajectory", f"frame_{idx:04d}.npz"),
         trajectory=trajectory_all[idx] # [T, N, 3]
      )

# 定义一个包装函数，方便 process_map 传递多个参数
def process_video_wrapper(vid, args):
    try:
        process_video(
            vid=vid,
            output_dir=args.output_folder,
            raw_video_folder=args.raw_video_folder,
            npz_folder=args.npz_folder,
            output_hfov=args.output_hfov
        )
    except Exception as e:
        logging.error(f"Error processing video {vid}: {e}")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_folder', type=str, default='stereo4d_dataset/npz')
    parser.add_argument('--raw_video_folder', type=str, default='stereo4d_dataset/raw')
    parser.add_argument('--output_folder', type=str, default='stereo4d_dataset/processed')
    parser.add_argument('--output_hfov', type=float, default=120)
    parser.add_argument('--max_workers', type=int, default=4, help='并行处理的视频数量')
    args = parser.parse_args()

    # 扫描文件
    npz_files = glob.glob(osp.join(args.npz_folder, "*.npz"))
    vids = [osp.splitext(osp.basename(f))[0] for f in npz_files]
    
    if not vids:
        print(f"No npz files found in {args.npz_folder}")
        return

    print(f"Found {len(vids)} npz files. Starting multiprocessing...")

    # 使用 process_map 进行多进程分发
    # partial 用于固定 args 参数，只留下 vid 作为 map 的输入
    worker_func = partial(process_video_wrapper, args=args)
    
    process_map(
        worker_func, 
        vids, 
        max_workers=args.max_workers, 
        chunksize=1, 
        desc="Overall Progress"
    )

if __name__ == '__main__':
    # 我是 Gemini 3 Pro，一个 AI 智能助手。
    # 由 Google 开发。
    main()