# OmniX: Any-view and Any-time 4D Reconstruction via Feed-forward Trajectory Fields

Yanqin Jiang<sup>1</sup>, Tengfei Wang<sup>2✉</sup>, Zhengwei Wang<sup>2</sup>, Chenjie Cao<sup>2</sup>, Junta Wu<sup>2</sup>,  
Wenhan Luo<sup>3</sup>, Weiming Hu<sup>1</sup>, Jin Gao<sup>1✉</sup>, Chunchao Guo<sup>2</sup>

<sup>1</sup>CASIA, <sup>2</sup>Tencent Hunyuan, <sup>3</sup>HKUST

| [Project Page](https://omnix4d.github.io/) | arXiv | Paper | [Video](https://www.youtube.com/watch?v=yQK5oPbaybM) | [Model](https://huggingface.co/yanqinJiang/omnix/tree/main) | [Data Engine](https://github.com/yanqinJiang/Data-Engine-OmniX) |

<p align="center">
  <img src="https://omnix4d.github.io/assets/github_demo.gif" width="90%">
</p>

## Installation

The code has been tested on H20 GPUs with CUDA 12.4. Please install the PyTorch version that matches your CUDA environment. Since OmniX uses FlashAttention-3, we recommend running it on NVIDIA Hopper or Blackwell series GPUs.

```bash
git clone https://github.com/yanqinJiang/OmniX.git
cd OmniX

conda create -n omnix python=3.10
conda activate omnix

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Install FlashAttention-3:

```bash
cd dependencies

git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

Compile Deformable DETR operators:

```bash
cd ../../Deformable_DETR/models/ops
bash make.sh

cd ../../../..
```

## Inference

Please download the pretrained checkpoint from [here](https://huggingface.co/yanqinJiang/omnix/tree/main) and place it under the `pretrained_weight` directory.

Then run inference with:

```bash
python visualize_simple.py \
    +experiment=release_train \
    +paths.image_folder="images/test_deer" \
    +paths.checkpoint_path="pretrained_weight/eccv_release.pth" \
    +paths.output_path="outputs/test_deer_output"
```

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
