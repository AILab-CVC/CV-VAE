<div align="center">
<h1>CV-VAE: A Compatible Video VAE for Latent 
Generative Video Models</h1>




</div>

> **TL; DR:** A video VAE for latent generative video models, which is compatible with pretrained image and video models, e.g., SD 2.1 and SVD

<p align="center">
  <img src="assets/i2v_and_t2v_results.gif">
</p>


## News

- [x] **2024-06-07** :hugs: We updated the text-to-image [inference code](sd21_vae3d_inference.ipynb) for SD2.1 + CV-VAE
- [x] **2024-06-03**  We have released the [inference code](cvvae_inference_video.py) and model weights of CV-VAE.


## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.13.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


### Video reconstruction


```bash
python3 cvvae_inference_video.py \
  --vae_path MODEL_PATH \
  --video_path INPUT_VIDEO_PATH \
  --save_path VIDEO_SAVE_PATH \
  --height HEIGHT \
  --width WIDTH 
```

