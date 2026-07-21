# OmniX for NVIDIA DGX Spark

This is the **DGX Spark fork** of [the original OmniX repository](https://github.com/yanqinJiang/OmniX). It keeps the upstream OmniX model and workflow while adding a tested, reproducible inference path for NVIDIA DGX Spark (ARM64/GB10).

## What this fork adds

- DGX Spark container builds with pinned NVIDIA PyTorch, CUDA, and Python dependencies.
- Checkpoint-compatible PyTorch scaled-dot-product attention in place of the Hopper-only FlashAttention-3 dependency.
- Deformable DETR CUDA operator builds for the GB10's `sm_120` architecture.
- Resumable long-video batch inference, validation, and high-resolution output workflows.
- Deterministic baking and verification of compact `.omx4d` files for browser playback.
- A browser-based 4D viewer for raw OmniX `.pt` predictions and baked `.omx4d` files.

## OMX4D viewer quickstart

The viewer opens `.omx4d` files directly in your browser; selected files stay local and are not uploaded. A sample file is included so you can try it immediately.

```bash
docker compose --file viewer/compose.yaml up --build
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173), then click **Open .pt / .omx4d** or drag an `.omx4d` file onto the page. Stop the viewer with:

```bash
docker compose --file viewer/compose.yaml down
```

For Node.js development, production builds, supported file details, and testing instructions, see the [viewer README](viewer/README.md).

---

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
- approximately 20 GiB free for the 9.1 GiB image, 4.9 GiB checkpoint, and the bundled example outputs; long-video batches need additional space described below
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

### Long-video batch inference

The creators train and evaluate with 16 frames at the native `504x280` model
resolution, and report that OmniX generalizes to 32 or more frames at inference
time. This fork validated 32-frame inference on the GB10, where it peaks at
29.69 GiB of allocated CUDA memory. For the dance footage below, 8 sampled
frames per second gave a useful four-second window per non-overlapping chunk.
A raw float32 PT shard is approximately 1.63 GiB; budget about 292.1 GiB for
179 shards before retaining videos, sampled frames, renders, or web assets.

Copy and edit the tracked example specification. Source videos and every
generated artifact stay under `outputs/`, which Git ignores:

```bash
cp configs/video_batch.example.json outputs/video_batch.json
# Edit source paths, IDs, output_root, and any source crop filter.

python3 scripts/prepare_video_batch.py \
  --spec outputs/video_batch.json

./scripts/run_spark_batch_inference.sh \
  outputs/video_batch_8fps_32f/batch_plan.json
```

Preparation uses FFmpeg from the pinned OmniX container, samples each source
once, and hard-links sampled JPEGs into non-overlapping 32-frame directories.
The final partial sequence repeats its last real frame to preserve a fixed
model shape; `valid_frames` and `pad_frames` in the manifest identify the
padding. The batch process verifies the source bytes and prepared chunk inputs,
loads the checkpoint once, runs chunks sequentially, and writes each PT
atomically. Re-running the same command verifies those inputs plus the recorded
crop, checkpoint, container, tensor shapes, fingerprint, and PT checksum before
skipping a completed shard.

Every filename remains meaningful when copied out of its directory, for
example:

```text
video_a__fps8__chunk-0000__t-000000000-000004000ms__valid32-pad00.pt
```

Verify all summaries, geometry checks, fingerprints, sizes, and hashes after a
batch finishes:

```bash
./scripts/run_spark_batch_verify.sh \
  outputs/video_batch_8fps_32f/batch_plan.json \
  --rehash
```

The DGX Spark acceptance batch used these three complete sources:

| Video | Sampled frames | PT shards | Final shard | Preprocessing |
| --- | ---: | ---: | --- | --- |
| TWICE “TT” | 2,028 | 64 | 12 valid + 20 padded | none |
| TWICE “Heart Shaker” | 1,532 | 48 | 28 valid + 4 padded | none |
| LOVE ATTACK 180 | 2,119 | 67 | 7 valid + 25 padded | left-eye crop: `crop=iw/2:ih/2:0:ih/4` |
| **Total** | **5,679** | **179** | **49 padded frames** | |

All 179 shards passed an aggregate acceptance run with fresh SHA-256
recomputation, exact raw tensor shape/dtype checks, complete finite-value scans,
camera-geometry checks, and contiguous non-overlap validation. The PTs occupy
313,635,259,997 bytes (292.10 GiB). Inference averaged 14.55 seconds per shard,
43.40 minutes total excluding model load and verification, and peaked at
29.69 GiB of allocated CUDA memory. Representative TT, Heart Shaker, and LOVE
ATTACK renders retained coherent backgrounds, recognizable dancers, temporal
motion tracks, and stable orbit views.

The manifest and per-chunk `status.json` files make the workflow resumable.
Use a new `output_root` when intentionally changing sampling or model settings;
fingerprint checks prevent old results from being silently accepted.

### High-resolution 500k OMX4D workflow

The batch specification also accepts `model_width` and `model_height`. Both
must be positive multiples of the model's 14-pixel patch size. The creators'
native setting remains `504x280`; `700x392` is an inference-time
generalization with nearly the same aspect ratio and a valid `50x28` patch
grid. Validate one representative shard before committing a multi-terabyte
run.

The three-video high-resolution production setting is:

```json
{
  "sampling_fps": 24,
  "frames_per_chunk": 32,
  "model_width": 700,
  "model_height": 392,
  "output_root": "outputs/youtube_omnix/full_24fps_32f_700x392"
}
```

At 24 fps the production run completed all three sources with 534
non-overlapping shards:

| Video | Sampled frames | PT/OMX4D shards | Final shard |
| --- | ---: | ---: | --- |
| TWICE “TT” | 6,085 | 191 | 5 valid + 27 padded |
| TWICE “Heart Shaker” | 4,597 | 144 | 21 valid + 11 padded |
| LOVE ATTACK 180 | 6,358 | 199 | 22 valid + 10 padded |
| **Total** | **17,040** | **534** | |

The accepted raw batch completed all 534 shards and passed fresh SHA-256
recomputation, exact tensor shape and dtype checks, finite-value scans,
camera-geometry checks, fingerprint validation, and contiguous non-overlap
validation. It wrote 1,819,314,920,780 bytes (1.655 TiB) of PT data. Recorded
shard inference totaled 19,201.22 seconds (5 hours 20 minutes), averaged 35.96
seconds per shard, and peaked at 52.99 GiB of allocated CUDA memory.

Before the full run, the LOVE ATTACK `chunk_0057` pilot completed inference in
55.37 seconds and produced a 3,406,956,889-byte PT with exact trajectory shape
`[32, 32, 392, 700, 3]`. All values and geometry checks passed. Its inspected
24 fps point-cloud render retained the five dancers, background, and temporal
motion.

After the raw batch passes `run_spark_batch_verify.sh --rehash`, bake one
OMX4D v1 file per PT:

```bash
./scripts/run_spark_omx4d_bake.sh \
  outputs/youtube_omnix/full_24fps_32f_700x392/batch_plan.json \
  --point-budget 500000 \
  --dynamic-reserved-fraction 0.8

./scripts/run_spark_omx4d_verify.sh \
  outputs/youtube_omnix/full_24fps_32f_700x392/batch_plan.json \
  outputs/youtube_omnix/full_24fps_32f_700x392/omx4d_500k_80d20s
```

The 500k budget is per shard. Selection is deterministic and identity-stable
across all 32 target frames:

- exactly 400,000 identities are the global highest dynamic scores across the
  valid source views; the threshold is zero, so selection continues into lower
  scores until the reserve is full;
- exactly 100,000 additional, non-duplicate identities are distributed through
  normalized frame-zero 3D voxels to retain scene coverage;
- padded source views in final shards are excluded from the candidate set;
- identities are ranked using the raw model scores; after selection, serialized
  float32 `dynamicScore` values are clamped to `[0, 1]` to absorb occasional
  few-ULP probability overshoots without changing point selection or ordering;
- source RGB, dynamic score, source-view index, camera poses, and intrinsics are
  stored alongside `float32 [32, 500000, 3]` positions.

The pilot OMX4D was 196,504,872 bytes. Its binary manifest reported the exact
400k/100k split, and an independent check proved that its highest 400,000
stored scores exactly matched the raw PT's global top 400,000, with cutoff
`0.966971457`. All 32 source views contributed, all positions were finite,
and recomputed bounds matched the manifest exactly.

The full bake produced 534 OMX4D files totaling 104,933,603,408 bytes (97.727
GiB): 191 for TT, 144 for Heart Shaker, and 199 for LOVE ATTACK. Exhaustive
validation rehashed all 534 files and checked the binary schema and descriptors,
the exact 500,000-point 400k/100k split, finite positions and calibration,
dynamic-score range, recomputed bounds, provenance, sidecars, and padded-source
exclusion before publishing the catalog. The production artifacts are organized
as follows:

```text
Batch plan/report:
outputs/youtube_omnix/full_24fps_32f_700x392/{batch_plan.json,batch_report.json}

Raw PT shards:
outputs/youtube_omnix/full_24fps_32f_700x392/output/<video>/chunk_*/

OMX4D files and sidecars:
outputs/youtube_omnix/full_24fps_32f_700x392/omx4d_500k_80d20s/<video>/chunk_*.{omx4d,json}

Validation report/catalog:
outputs/youtube_omnix/full_24fps_32f_700x392/omx4d_500k_80d20s/{validation_report.json,catalog.json}
```

The exhaustive verifier rehashes every OMX4D, scans every position and
calibration array, checks padding exclusion and provenance, and publishes
`catalog.json` only if the complete file set passes. OMX4D chunks retain
stable point identities within a chunk, not across chunks; a web player should
stream the catalog in sequence and reset or crossfade at boundaries.

### Browser timeline package

Raw PTs are deliberately preserved, but a browser should not download a
1.63 GiB tensor for each four-second chunk. Create a complete web package with:

```bash
./scripts/run_spark_web_pack.sh \
  outputs/video_batch_8fps_32f/batch_plan.json \
  --strict
```

For each chunk, the packer chooses one middle reference cloud and a deterministic
mixture of 8,192 raster-distributed and high-dynamic-score points. It writes:

- positions as little-endian float32 `[time, point, xyz]`
- pixel indices, RGB colors, and quantized dynamic scores
- per-frame camera-to-world matrices and intrinsics
- SHA-256 and shape/axis metadata for every binary

The accepted three-video package contains 1,074 binary files totaling
575,297,408 bytes (548.65 MiB), plus its manifest. The browser loader
checksum-validated real first, middle, and final shards for all three videos,
including exact four-second boundaries and the short unsampled source tail.

The position stream is 3,145,728 bytes per 32-frame chunk, over 500 times
smaller than its raw PT. The complete package is written to
`<output_root>/web/manifest.json`; filtered or incomplete packing runs write
`manifest.partial.json` and never replace the canonical timeline. Serve the
directory over HTTP(S), rather than opening it with `file://`, so browser fetch
and Web Crypto checksum verification are available.

[examples/omnix_timeline_loader.js](examples/omnix_timeline_loader.js) provides
a dependency-free browser loader with validation, checksum checking, gap-safe
timeline lookup, and a small lazy cache:

```js
import { openOmniXTimeline } from "./omnix_timeline_loader.js";

const timeline = await openOmniXTimeline("/omnix/manifest.json");
const location = timeline.locate("video_a", 42.3);
const chunk = await timeline.loadChunk(location.videoId, location.chunkId);
const xyz = chunk.framePositions(location.frameIndex);
```

The PTs cannot be physically concatenated as one continuous 4D world. Each
independent inference chunk has its own normalized scale and coordinate gauge,
and its reference pixels are different point identities. The web manifest is a
logical timeline: prefetch the next shard, render one chunk at a time, and reset
the scene or crossfade rendered images in screen space at its boundary. Do not
interpolate unaligned point clouds. A future continuous-world stitch should
estimate a robust boundary similarity transform from static-background 2D
matches and their predicted 3D points, and start a new segment at hard cuts or
weak matches.

OmniX matrices use an OpenCV camera basis (`+x` right, `+y` down, `+z`
forward). For Three.js/WebGL, with `C = diag(1, -1, -1, 1)`, transform points as
`p_three = C * p_opencv` and homogeneous camera-to-world matrices as
`c2w_three = C * c2w_opencv * C`. The web manifest records this convention.

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
