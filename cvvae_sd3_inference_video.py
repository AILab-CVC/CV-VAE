from models.modeling_vae import CVVAESD3Model
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire


def main(vae_path, video_path, save_path, height=576, width=1024):
    vae3d = CVVAESD3Model.from_pretrained(
        vae_path, subfolder="vae3d_sd3", torch_dtype=torch.float16
    )
    vae3d.requires_grad_(False)

    transform = transforms.Compose([transforms.Resize(size=(height, width))])

    vae3d = vae3d.cuda()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video_reader = VideoReader(video_path, ctx=cpu(0))

    fps = video_reader.get_avg_fps()

    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()

    video = rearrange(torch.tensor(video), "t h w c -> t c h w")

    video = transform(video)

    video = rearrange(video, "t c h w -> c t h w").unsqueeze(0).half()

    frame_end = 1 + (len(video_reader) - 1) // 4 * 4

    video = video / 127.5 - 1.0

    video = video[:, :, :frame_end, :, :]

    video = video.cuda()

    print(f"Shape of input video: {video.shape}")
    latent = vae3d.encode(video).latent_dist.sample()

    print(f"Shape of video latent: {latent.shape}")

    results = vae3d.decode(latent).sample

    results = rearrange(results.squeeze(0), "c t h w -> t h w c")

    results = (torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5
    results = results.to("cpu", dtype=torch.uint8)

    write_video(save_path, results, fps=fps, options={"crf": "10"})


if __name__ == "__main__":
    Fire(main)
