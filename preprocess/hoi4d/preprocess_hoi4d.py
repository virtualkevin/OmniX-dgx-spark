import os
import json
import torch
import cv2
import numpy as np
import shutil
import argparse
import sys
from multiprocessing import Pool
from functools import partial
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess HOI4D Scene Data with optional Resizing (multi-scene, multi-process)"
    )

    # Path Arguments
    parser.add_argument("--annotation_root", type=str, required=True,
                        help="Root directory for annotations (contains scene folders)")
    parser.add_argument("--raw_data_root", type=str, required=True,
                        help="Root directory for raw data (contains video and depth folders)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Directory to save processed data")

    # Optional Arguments
    parser.add_argument("--resize_height", type=int, default=None,
                        help="Target height for RGB and Depth images. Maintains aspect ratio. If not set, original size kept.")

    # Parallelism / selection
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes (scenes processed in parallel).")
    parser.add_argument("--scene_limit", type=int, default=None,
                        help="Only process first N scenes (for debugging).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a scene if output folder already exists.")
    parser.add_argument("--quiet", action="store_true",
                        help="Less per-frame printing; only per-scene logs.")

    # New: failure log path
    parser.add_argument(
        "--fail_log", type=str, default=None,
        help="Path to save failed scenes log (tsv). Default: <output_root>/failed_scenes_<timestamp>.tsv"
    )

    return parser.parse_args()


def load_camera_info(info_json_path: str):
    if not os.path.exists(info_json_path):
        raise FileNotFoundError(f"Info file not found: {info_json_path}")

    with open(info_json_path, "r") as f:
        info_data = json.load(f)

    extrinsics = torch.tensor(info_data["extrinsics"])
    num_frames = extrinsics.shape[0]

    if "crop_intrinsic" in info_data:
        fx, fy, cx, cy = info_data["crop_intrinsic"].values()
    elif "intrinsic" in info_data:
        fx, fy, cx, cy = info_data["intrinsic"].values()
    else:
        fx, fy, cx, cy = 0, 0, 0, 0
        print("Warning: Intrinsic key not found, using zeros.")

    intrinsic = torch.eye(3)
    intrinsic[0, 0] = fx
    intrinsic[0, 2] = cx
    intrinsic[1, 1] = fy
    intrinsic[1, 2] = cy

    intrinsics = intrinsic.unsqueeze(0).repeat(num_frames, 1, 1)
    return intrinsics, extrinsics


def list_scenes(annotation_root: str):
    if not os.path.isdir(annotation_root):
        raise ValueError(f"annotation_root is not a directory: {annotation_root}")

    scenes = []
    for name in os.listdir(annotation_root):
        p = os.path.join(annotation_root, name)
        if os.path.isdir(p):
            scenes.append(name)

    scenes.sort()
    return scenes


def preprocess_one_scene(scene_name: str, args):
    info_path = os.path.join(args.annotation_root, scene_name, "camera", "recon", "split_0", "info.json")
    image_list_path = os.path.join(args.annotation_root, scene_name, "camera", "image_list.json")
    video_path = os.path.join(args.raw_data_root, scene_name.replace("_", "/"), "align_rgb", "image.mp4")
    depth_folder_src = os.path.join(args.annotation_root, scene_name, "prior_depth")

    scene_output_dir = os.path.join(args.output_root, scene_name)
    out_img_dir = os.path.join(scene_output_dir, "image")
    out_depth_dir = os.path.join(scene_output_dir, "depth")
    out_cam_dir = os.path.join(scene_output_dir, "camera")

    if args.skip_existing and os.path.isdir(scene_output_dir):
        return (scene_name, "SKIP_EXISTING", None)

    if not os.path.exists(image_list_path):
        return (scene_name, "ERROR_IMAGE_LIST_MISSING", image_list_path)

    with open(image_list_path, "r") as file:
        image_names = json.load(file)

    image_names_without_ext = [name.split(".")[0] for name in image_names]
    num_meta_frames = len(image_names)

    try:
        intrinsics_torch, extrinsics_w2c_torch = load_camera_info(info_path)
    except Exception as e:
        return (scene_name, "ERROR_CAMERA_INFO", str(e))

    if not os.path.exists(video_path):
        return (scene_name, "ERROR_VIDEO_MISSING", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (scene_name, "ERROR_VIDEO_OPEN_FAIL", video_path)

    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if num_video_frames != num_meta_frames:
        cap.release()
        return (scene_name, "ERROR_FRAMECOUNT_MISMATCH", f"meta={num_meta_frames}, video={num_video_frames}")

    if extrinsics_w2c_torch.shape[0] != num_meta_frames:
        cap.release()
        return (scene_name, "ERROR_EXTRINSICS_MISMATCH", f"extr={extrinsics_w2c_torch.shape[0]}, meta={num_meta_frames}")

    resize_scale = 1.0
    target_w, target_h = orig_w, orig_h

    if args.resize_height is not None and args.resize_height < orig_h:
        resize_scale = args.resize_height / float(orig_h)
        target_h = args.resize_height
        target_w = int(orig_w * resize_scale)
    elif args.resize_height is not None and args.resize_height >= orig_h:
        resize_scale = 1.0
        target_w, target_h = orig_w, orig_h

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    for idx in range(num_video_frames):
        ret, frame = cap.read()
        if not ret:
            break

        original_name = image_names_without_ext[idx]
        target_img_name = f"frame_{idx:04d}.jpg"
        target_depth_name = f"frame_{idx:04d}.png"
        target_cam_name = f"frame_{idx:04d}.npz"

        if resize_scale != 1.0:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_img_dir, target_img_name), frame)

        src_depth_path = os.path.join(depth_folder_src, original_name + ".png")
        dst_depth_path = os.path.join(out_depth_dir, target_depth_name)

        if os.path.exists(src_depth_path):
            if resize_scale != 1.0:
                depth_map = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
                if depth_map is not None:
                    depth_resized = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(dst_depth_path, depth_resized)
            else:
                shutil.copy2(src_depth_path, dst_depth_path)
        else:
            if not args.quiet:
                print(f"[{scene_name}] Warning: Depth missing for frame {idx}: {src_depth_path}")

        K = intrinsics_torch[idx].numpy().astype(np.float32)
        w2c = extrinsics_w2c_torch[idx].numpy().astype(np.float32)

        if resize_scale != 1.0:
            K[0, 0] *= resize_scale
            K[1, 1] *= resize_scale
            K[0, 2] *= resize_scale
            K[1, 2] *= resize_scale

        if w2c.shape == (4, 4):
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
        else:
            c2w = np.eye(4, dtype=np.float32)

        np.savez(
            os.path.join(out_cam_dir, target_cam_name),
            intrinsic=K,
            camera_pose=c2w,
        )

        if (not args.quiet) and idx % 200 == 0:
            sys.stdout.write(f"\r[{scene_name}] Processed {idx}/{num_video_frames} frames...")
            sys.stdout.flush()

    cap.release()
    if not args.quiet:
        print(f"\n[{scene_name}] Done: {scene_output_dir}")

    return (scene_name, "OK", scene_output_dir)


def _worker_init():
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _write_fail_log(fail_log_path: str, results):
    # results: list of (scene_name, status, info)
    fails = [(s, st, info) for (s, st, info) in results if st not in ("OK", "SKIP_EXISTING")]
    if not fails:
        print("No failed scenes; fail log not written.")
        return

    os.makedirs(os.path.dirname(fail_log_path), exist_ok=True)

    with open(fail_log_path, "w", newline="") as f:
        f.write("scene_name	status	info\n")
        for scene_name, status, info in fails:
            # info 里可能有换行，简单做一下清理，避免 tsv 断行
            info_str = "" if info is None else str(info).replace("\n", "\\n").replace("\t", " ")
            f.write(f"{scene_name}\t{status}\t{info_str}\n")

    print(f"Failed scenes log saved to: {fail_log_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    if args.fail_log is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.fail_log = os.path.join(args.output_root, f"failed_scenes_{ts}.tsv")

    scenes = list_scenes(args.annotation_root)
    # # tmp_debug
    # with open("/apdcephfs/private_yanqinjiang/project/dream4d/preprocess/hoi4d/fail_scene.txt", "r") as file_to_read:
    #     lines = file_to_read.readlines()[1:]
    # scenes = [line.strip().split("	")[0] for line in lines]
    if args.scene_limit is not None:
        scenes = scenes[:args.scene_limit]
        # scenes = ["ZY20210800001_H1_C1_N19_S100_s02_T1"]

    print(f"Found {len(scenes)} scenes under: {args.annotation_root}")
    print(f"Using {args.num_workers} workers")
    if args.resize_height:
        print(f"Resize height: {args.resize_height}")
    else:
        print("Resize: disabled")
    print(f"Fail log: {args.fail_log}")

    worker_fn = partial(preprocess_one_scene, args=args)

    results = []
    with Pool(processes=args.num_workers, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(worker_fn, scenes):
            scene_name, status, info = r
            results.append(r)
            if status == "OK":
                print(f"[OK]   {scene_name}")
            elif status == "SKIP_EXISTING":
                print(f"[SKIP] {scene_name} (output exists)")
            else:
                print(f"[FAIL] {scene_name} -> {status}: {info}")

    ok = sum(1 for _, s, _ in results if s == "OK")
    skip = sum(1 for _, s, _ in results if s == "SKIP_EXISTING")
    fail = len(results) - ok - skip

    print("====================================")
    print(f"Done. OK={ok}, SKIP={skip}, FAIL={fail}")
    print("====================================")

    _write_fail_log(args.fail_log, results)


if __name__ == "__main__":
    main()