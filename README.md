# StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References
This is the official PyTorch implementation of the following publication: 

> **"StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References".** <br>
> Boyu He, Yunfan Ye, Chang Liu, Wuwei Shang, Liu Fang, Zhiping Cai <br>
> *CVPR 2026* <br>
> [Project Page]() | [Paper]()

## 📖 Introduction
**TL;DR: StyleGallery is a training-free and semantic-aware framework to generate a high-quality stylized image from arbitrary image references.** 
<div align="center">
  <img src="assets/teaser.png" width="800">
</div>


<P align="justify">
<strong>Abstract:</strong> Despite the advancements in diffusion-based image style transfer, existing methods are commonly limited by 1) semantic gap: the style reference could miss proper content semantics, causing uncontrollable stylization; 2) reliance on extra constraints (e.g., semantic masks) restricting applicability; 3) rigid feature associations lacking adaptive global-local alignment, failing to balance fine-grained stylization and global content preservation. These limitations, particularly the inability to flexibly leverage style inputs, fundamentally restrict style transfer in terms of personalization, accuracy, and adaptability. To address these, we propose StyleGallery, a training-free and semantic-aware framework that supports arbitrary reference images as input and enables effective personalized customization. It comprises three core stages: semantic region segmentation (adaptive clustering on latent diffusion features to divide regions without extra inputs); clustered region matching (block filtering on extracted features for precise alignment); and style transfer optimization (energy function–guided diffusion sampling with regional style loss to optimize stylization). Experiments on our introduced benchmark demonstrate that StyleGallery outperforms state-of-the-art methods in content structure preservation, regional stylization, interpretability, and personalized customization, particularly when leveraging multiple style references.
</P>

## 🚀 Setup
The code has been tested on:

* Ubuntu 20.04 
* CUDA 12.6
* Python 3.10.18
* Pytorch 2.6.0
* GeForce RTX 4090.

### 1. Create a Conda Environment
```bash
# create env using conda
conda create -n StyleGallery python=3.10
conda activate StyleGallery

# install dependencies with pip
pip install -r requirements.txt
```

### 2. Download pretrained models
StyleGallery is training-free but utilizes pretrained models of several existing projects. 
For the basic running, we need to download [SD1.5](https://ai.gitee.com/hf-models/runwayml/stable-diffusion-v1-5/tree/main) and [dinov2](https://huggingface.co/facebook/dinov2-base).
