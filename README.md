# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging
<div align="center">
<p style='text-align: center;'>
            <strong>V0.1, Introduced By 
            <a href="https://gaplab.cuhk.edu.cn/" target="_blank">GAP Lab</a> from CUHKSZ and 
            <a href="https://www.nvsgames.cn/" target="_blank">Game-AIGC Team</a> from ByteDance</strong>
</p>
</div>

![image](https://github.com/user-attachments/assets/7170101f-fc40-44f3-912e-534d10d6804e)
 
<div align="center">

[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://stable-x.github.io/Hi3DGen/) 
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2503.22236) 
[![Online Demo](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/Hi3DGen) 
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/trellis-normal-v0-1) 
 </div>

Hi3DGen target at generating high-fidelity 3D geometry from images using normal maps as an intermediate representation. The framework addresses limitations in existing methods that struggle to reproduce fine-grained geometric details from 2D inputs.

## News
**[April 13, 2025]** 🚀 Hi3DGen v0.1 Released! 🔥🔥🔥 Hi3DGen-MV is still under training, we expect to release it by April 20th. Model Offloading for compatible 8 GB GPU memory is expected by April 17th.

## Installation
Clone the repo:
```bash
git clone --recursive https://github.com/ByteDance/Hi3DGen.git
cd Hi3DGen
```

Create a conda environment (optional):
```bash
conda create -n stablex python=3.10
conda activate stablex
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/{your-cuda-version}
pip install spconv-cu{your-cuda-version}==2.3.6 xformers==0.0.27.post2
# other dependencies
pip install -r requirements.txt
```

## Local Demo 🤗
Run by:
```bash
python app.py
```

<!-- License -->
## License
The model and code of Hi3DGen are adapted from [**Trellis**](https://github.com/microsoft/TRELLIS), which are licensed under the [MIT License](LICENSE). While the original Trellis is MIT licensed, we have specifically removed its dependencies on certain NVIDIA libraries (kaolin, nvdiffrast, flexicube) to ensure this adapted version can be used commercially. Hi3DGen itself is distributed under the [MIT License](LICENSE).

## Citation
If you find this work helpful, please consider citing our paper:
```
@article{ye2025hi3dgen,
  title={Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging},
  author={Ye, Chongjie and Wu, Yushuang and Lu, Ziteng and Chang, Jiahao and Guo, Xiaoyang and Zhou, Jiaqing and Zhao, Hao and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2503.22236}, 
  year={2025}
}
```
