# OmniX: Any-view and Any-time 4D Reconstruction via Feed-forward Trajectory Fields

Yanqin Jiang<sup>1</sup>, Tengfei Wang<sup>2✉</sup>, Zhengwei Wang<sup>2</sup>, Chenjie Cao<sup>2</sup>, Junta Wu<sup>2</sup>,  
Wenhan Luo<sup>3</sup>, Weiming Hu<sup>1</sup>, Jin Gao<sup>1✉</sup>, Chunchao Guo<sup>2</sup>

<sup>1</sup>CASIA, <sup>2</sup>Tencent Hunyuan, <sup>3</sup>HKUST

| [Project Page](https://omnix4d.github.io/) | arXiv | Paper | [Video](https://www.youtube.com/watch?v=yQK5oPbaybM) | [Model](https://huggingface.co/yanqinJiang/omnix/tree/main) | [Data Engine](https://github.com/yanqinJiang/Data-Engine-OmniX) |

<p align="center">
  <img src="https://omnix4d.github.io/assets/github_demo.gif" width="90%">
</p>

## DGX Spark inference

This fork includes a tested ARM64/GB10 inference path for DGX Spark. It replaces the Hopper-only FlashAttention-3 dependency with checkpoint-compatible PyTorch scaled-dot-product attention and compiles the Deformable DETR CUDA operator for `sm_120`.

The container is reproducibly pinned to:

- NVIDIA PyTorch `25.11-py3` by image digest (PyTorch 2.10, CUDA 13.0)
- Depth Anything 3 commit `e74fd796e96b7e781a5506fd8503b6bd7232513c`
- the Python versions in `requirements-spark.txt`

### Prerequisites

- DGX Spark with Docker and NVIDIA Container Toolkit configured
- approximately 20 GiB free for the 9.1 GiB image, 4.9 GiB checkpoint, and outputs
- internet access while building the image and downloading the checkpoint

Confirm that Docker can see the GB10:

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 \
  python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

Expected device output is `NVIDIA GB10 (12, 1)`.

### Download the release checkpoint

The release file is named `eccv_release.ckpt` (not `.pth`). Install the current Hugging Face CLI if `hf` is not already available, then download it:

```bash
python3 -m pip install --user -U huggingface_hub
mkdir -p pretrained_weight
hf download yanqinJiang/omnix eccv_release.ckpt \
  --local-dir pretrained_weight
```

Verify the exact file before inference:

```bash
echo "871638d02bf8591ccb24de9da28bf862a0c1fa10a8640442e76cb58face343ed  pretrained_weight/eccv_release.ckpt" \
  | sha256sum --check
```

The checkpoint is 5,306,117,122 bytes. Checkpoints, generated outputs, native build products, and caches are excluded from Git and the Docker build context.

### Build and run

```bash
./scripts/build_spark_image.sh
./scripts/run_spark_inference.sh
```

The run script processes all 16 bundled images under `images/test_deer`, validates the output tensors and camera geometry, and writes:

- `outputs/test_deer_output/prediction_summary.json` — shapes, ranges, finite checks, geometry checks, motion statistics, and peak allocated CUDA memory
- `outputs/test_deer_output/predictions.pt` — raw float32 trajectory, pose, intrinsics, and dynamic-score tensors
- `outputs/test_deer_output/video_cam_00.mp4` through `video_cam_15.mp4` — temporal point-cloud and trajectory renders
- `outputs/test_deer_output/gt_images/` — the cropped input frames used by inference

For a fast two-image numerical smoke test without raw tensors or video rendering:

```bash
./scripts/run_spark_inference.sh \
  ++paths.output_path=outputs/test_deer_smoke \
  +paths.max_images=2 \
  +paths.skip_render=true \
  +paths.save_raw_predictions=false
```

`++` is intentional when replacing a default path already supplied by the script. New options use a single `+`.

### Validated deer result

The full 16-image example was run successfully on a DGX Spark GB10. The checkpoint matched all 1,119 network tensors exactly, and the following acceptance checks passed:

| Check | Observed result |
| --- | --- |
| Required outputs | trajectory, camera pose, intrinsics, dynamic score |
| Finite values | 100% for every required tensor |
| Trajectory shape | `[1, 16, 16, 280, 504, 3]` |
| Camera rotation max orthogonality error | `1.79e-7` |
| Camera rotation determinant range | `0.9999998` to `1.0000001` |
| Dynamic-score range | `0.0` to `0.9684` |
| Pixels above dynamic score 0.5 | `5.64%` |
| Peak allocated CUDA memory | `12.46 GiB` |

Representative renders were also inspected: the street geometry is coherent, the deer silhouette is recognizable, and colored motion tracks follow the deer through the temporal and orbit frames rather than producing blank, exploded, or non-finite output.

### Troubleshooting

- `docker: Error response ... could not select device driver`: install or repair NVIDIA Container Toolkit, then rerun the GPU check above.
- `Missing checkpoint`: confirm the filename is exactly `pretrained_weight/eccv_release.ckpt` and verify its SHA-256.
- Native operator import errors: rebuild with `./scripts/build_spark_image.sh`; do not reuse the checked-in x86/Python 3.10 build artifacts.
- For a smaller diagnostic run, use `+paths.max_images=2`, `+paths.skip_render=true`, and `+paths.save_raw_predictions=false`.
- To use another local image layout, call `visualize_simple.py` with Hydra path overrides. It accepts either one folder of images or subfolders representing multiple videos.

## Conventional installation

The original project was tested on H20 GPUs with CUDA 12.4. For training or a non-Spark environment, install the PyTorch build matching that system, install `requirements.txt`, and compile `dependencies/Deformable_DETR/models/ops` locally. The DGX Spark container above is the supported inference path for this fork.

## Training

We provide dataset preprocessing code in the `preprocess` folder. For DL3DV and Spring datasets, please refer to the preprocessing pipeline of [CUT3R](https://github.com/CUT3R/CUT3R).

Please download the [Depth Anything 3 checkpoint](https://huggingface.co/spaces/depth-anything/depth-anything-3) and place it under the `pretrained_weight` directory. Then use `pretrained_weight/convert_pt.py` to convert the DA3 checkpoint into the format used by this repository as initialization for our model.

Start training with:

```bash
python src/train.py +experiment=release_train
```

Note that the provided training configuration is for reference and may need to be adjusted according to your hardware and dataset setup.

## Acknowledgements

This project builds upon several excellent open-source projects and research efforts. We sincerely thank the authors and contributors of [CUT3R](https://github.com/CUT3R/CUT3R), [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3), [VGGT](https://github.com/facebookresearch/vggt), and [WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) for their inspiring works, released models, codebases, and resources.

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{jiang2026omnix,
  title     = {OmniX: Any-view and Any-time 4D Reconstruction via Feed-forward Trajectory Fields},
  author    = {Jiang, Yanqin and Wang, Tengfei and Wang, Zhengwei and Cao, Chenjie and Wu, Junta and Luo, Wenhan and Hu, Weiming and Gao, Jin and Guo, Chunchao},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
