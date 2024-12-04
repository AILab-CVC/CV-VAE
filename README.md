<div align="center">
<h1>CV-VAE: A Compatible Video VAE for Latent 
Generative Video Models</h1>

[Sijie Zhao](https://scholar.google.com/citations?user=tZ3dS3MAAAAJ) · [Yong Zhang*](https://yzhang2016.github.io/) · [Xiaodong Cun](https://vinthony.github.io/academic/) · [Shaoshu Yang]() · [Muyao Niu]()

[Xiaoyu Li](https://xiaoyu258.github.io/) · [Wenbo Hu](https://wbhu.github.io/) · [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)

<sup>*</sup>Corresponding Authors


<a href='https://ailab-cvc.github.io/cvvae/index.html'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2405.20279'><img src='https://img.shields.io/badge/Technique-Report-red'></a>


</div>

> **TL; DR:** A video VAE for latent generative video models, which is compatible with pretrained image and video models, e.g., SD 2.1 and SVD

<p align="center">
  <img src="assets/i2v_and_t2v_results.gif">
</p>


## News

- [x] **2024-10-14** :hugs: We have updated the training code of CV-VAE.
- [x] **2024-10-14**  We have released the [inference code](cvvae_sd3_inference_video.py) and [model weights](https://huggingface.co/AILab-CVC/CV-VAE/tree/main/vae3d_sd3) of CV-VAE-SD3 which is compatible with [SD3](stabilityai/stable-diffusion-3-medium) and [SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).
- [x] **2024-10-14**  We have updated the CV-VAE with better performance, please check [cv-vae-v1-1](https://huggingface.co/AILab-CVC/CV-VAE/tree/main/vae3d_v1-1).
- [x] **2024-09-25**  CV-VAE is accepted by NeurIPS 2024.
- [x] **2024-06-03**  We have released the [inference code](cvvae_inference_video.py) and [model weights](https://huggingface.co/AILab-CVC/CV-VAE/tree/main/vae3d) of CV-VAE.

- [x] **2024-05-30**  We have updated the [arXiv preprint](https://arxiv.org/abs/2405.20279).

## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.13.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


### Video reconstruction

Download the model weight from [Hugging Face](https://huggingface.co/AILab-CVC/CV-VAE/tree/main)

```bash
python3 cvvae_inference_video.py \
  --vae_path MODEL_PATH \
  --video_path INPUT_VIDEO_PATH \
  --save_path VIDEO_SAVE_PATH \
  --height HEIGHT \
  --width WIDTH 
```


## 😉 Citation
```
@article{zhao2024cvvae,
  title={CV-VAE: A Compatible Video VAE for Latent Generative Video Models},
  author={Zhao, Sijie and Zhang, Yong and Cun, Xiaodong and Yang, Shaoshu and Niu, Muyao and Li, Xiaoyu and Hu, Wenbo and Shan, Ying},
  journal={https://arxiv.org/abs/2405.20279},
  year={2024}
}
```
