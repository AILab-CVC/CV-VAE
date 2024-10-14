import os
import random
from tqdm import tqdm
import json
import torch
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from safetensors import safe_open
from einops import rearrange, repeat
from safetensors.torch import load_file
import tempfile
import numpy as np
from torchvision.io import VideoReader as TorchVideoReader


def webvid_decoder(
    sample,
    data_root,
    video_length=16,
    resolution=[256, 512],
    spatial_transform=None,
    load_raw_resolution=False,
    frame_stride_range=[1, 8],
    cond_noise_range=[0.00, 0.04],  # deprecated
    cond_noise_log_mean_std=[-3.0, 0.5],
    random_start_frame=True,
    add_noise_to_cond=True,
    rank_time_dim_to_0=True,
):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
    try:
        if "page_dir" in sample and "videoid" in sample:
            video_path = os.path.join(
                data_root, "videos", sample["page_dir"], str(sample["videoid"]) + ".mp4"
            )
            caption = sample["name"]
        elif "filepath" in sample:
            video_path = os.path.join(data_root, sample["filepath"])
            caption = sample.get("caption", "")
        else:
            raise ValueError("Cannot parse video path from metadata!")
        if load_raw_resolution:
            video_reader = VideoReader(video_path, ctx=cpu(0))
        else:
            video_reader = VideoReader(
                video_path,
                ctx=cpu(0),
                width=resolution[1],
                height=resolution[0],
            )

        if len(video_reader) < video_length:
            print(
                f"video length ({len(video_reader)}) is smaller than target length({video_length})"
            )
            return {}
        fps_ori = video_reader.get_avg_fps()

        frame_stride = random.randint(frame_stride_range[0], frame_stride_range[1])

        ## get valid range (adapting case by case)
        required_frame_num = frame_stride * (video_length - 1) + 1
        frame_num = len(video_reader)

        if frame_num < video_length:
            return {}
        elif frame_num < required_frame_num:
            frame_stride = frame_num // video_length
            required_frame_num = frame_stride * (video_length - 1) + 1

        fps_clip = round(fps_ori / frame_stride)

        ## select a random clip
        random_range = frame_num - required_frame_num
        if random_start_frame:
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
        else:
            start_idx = 0
        frame_indices = [start_idx + frame_stride * i for i in range(video_length)]

        frames = video_reader.get_batch(frame_indices)

        assert (
            frames.shape[0] == video_length
        ), f"{len(frames)}, self.video_length={video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
        )  # [t,h,w,c] -> [t,c,h,w]
        if spatial_transform is not None:
            frames = spatial_transform(frames)
        # if resolution is not None:
        #     assert (frames.shape[2], frames.shape[3]) == (
        #         resolution[0],
        #         resolution[1],
        #     ), f"frames={frames.shape}, self.resolution={resolution}"
        frames = (frames / 255 - 0.5) * 2
        # cond_noise_aug = random.uniform(cond_noise_range[0], cond_noise_range[1])
        if add_noise_to_cond:
            cond_noise_aug = np.exp(
                cond_noise_log_mean_std[0]
                + cond_noise_log_mean_std[1] * np.random.normal()
            )
        else:
            cond_noise_aug = 0.0

        cond_frames = frames[0] + cond_noise_aug * torch.randn_like(frames[0])

        if not rank_time_dim_to_0:
            input_frames = frames.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
        else:
            input_frames = frames

        data = {
            "frames": input_frames,
            "cond_frames_without_noise": frames[0],
            "fps_id": torch.tensor(fps_clip),
            "motion_bucket_id": torch.tensor(127),
            "cond_frames": cond_frames,
            "cond_aug": torch.tensor(cond_noise_aug),
            "num_video_frames": video_length,
            "caption": caption,
            "path": video_path,
            "frame_stride": frame_stride,
            "image_only_indicator": torch.zeros(video_length),
        }

        return data

    except Exception as e:
        print(
            f"Error while decode image: {e}, abort!",
        )
        return {}


def csv_image_decoder(
    sample,
    data_root,
    spatial_transform=None,
    add_time_dim=True,
):
    try:
        filename = sample["filename"]
        filepath = os.path.join(data_root, filename)
        image = Image.open(filepath).convert("RGB")
        caption = sample.get("caption", "")

        if spatial_transform is not None:
            image = spatial_transform(image)
            image = (image - 0.5) * 2

        if add_time_dim:
            image = torch.unsqueeze(image, 1)

        return {"frames": image, "path": filepath, "caption": caption}

    except Exception as e:
        print(
            f"Error while decode image: {e}, abort!",
        )
        return {}


def webdata_image_decoder(
    sample, spatial_transform=None, load_txt=False, load_json=False, add_time_dim=True
):
    key, data = sample
    if key.endswith(".txt") and load_txt:
        caption = data.read().decode("utf-8")
        return key, {"caption": caption}
    elif key.endswith(".jpg"):
        try:
            image = Image.open(data).convert("RGB")
            if spatial_transform is not None:
                image = spatial_transform(image)
                image = (image - 0.5) * 2
        except Exception as e:
            print(f"Error while decode image from tar: {e}")
            return key, {}

        if add_time_dim:
            image = torch.unsqueeze(image, 1)

        return key, {"frames": image}
    elif key.endswith(".json") and load_json:
        try:
            metadata_str = data.read().decode("utf-8")
            metadata = json.loads(metadata_str)
        except Exception as e:
            print(f"Error while decode json from tar: {e}")
            return key, {}
        return key, {"metadata": metadata_str}
    else:
        return key, {}


def webdata_video_decoder(
    sample,
    video_length=16,
    # resolution=[256, 512],
    frame_stride_range=[1, 8],
    spatial_transform=None,
    random_start_frame=True,
    load_txt=False,
    load_json=False,
    rank_time_dim_to_0=True,
):
    key, data = sample
    if key.endswith(".txt") and load_txt:
        caption = data.read().decode("utf-8")
        return key, {"caption": caption}
    elif key.endswith(".mp4"):

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
                temp_file.write(data.read())
                temp_file.flush()

                video_reader = VideoReader(temp_file.name)
                if len(video_reader) < video_length:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({video_length})"
                    )
                    return {}
                fps_ori = video_reader.get_avg_fps()

                frame_stride = random.randint(
                    frame_stride_range[0], frame_stride_range[1]
                )

                ## get valid range (adapting case by case)
                required_frame_num = frame_stride * (video_length - 1) + 1
                frame_num = len(video_reader)

                if frame_num < video_length:
                    return {}
                elif frame_num < required_frame_num:
                    frame_stride = frame_num // video_length
                    required_frame_num = frame_stride * (video_length - 1) + 1

                fps_clip = round(fps_ori / frame_stride)

                ## select a random clip
                random_range = frame_num - required_frame_num
                if random_start_frame:
                    start_idx = (
                        random.randint(0, random_range) if random_range > 0 else 0
                    )
                else:
                    start_idx = 0
                frame_indices = [
                    start_idx + frame_stride * i for i in range(video_length)
                ]

                frames = video_reader.get_batch(frame_indices)
            assert (
                frames.shape[0] == video_length
            ), f"{len(frames)}, self.video_length={video_length}"

            frames = (
                torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
            )  # [t,h,w,c] -> [t,c,h,w]
            if spatial_transform is not None:
                frames = spatial_transform(frames)

            frames = (frames / 255 - 0.5) * 2

            if not rank_time_dim_to_0:
                input_frames = frames.permute(
                    1, 0, 2, 3
                )  # [t, c, h, w] -> [c, t, h, w]
            else:
                input_frames = frames

            return key, {
                "frames": input_frames,
            }

        except Exception as e:
            print(
                f"Error while decode video: {e}, abort!",
            )
            return key, {}

    elif key.endswith(".json") and load_json:
        try:

            metadata_str = data.read().decode("utf-8")
            metadata = json.loads(metadata_str)
        except Exception as e:
            print(f"Error while decode json from tar: {e}")
            return key, {}
        return key, {"metadata": metadata_str}
    else:
        return key, {}
