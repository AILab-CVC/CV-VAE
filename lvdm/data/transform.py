import torch

from torchvision import transforms
from torchvision.transforms.functional import (
    _interpolation_modes_from_int,
    InterpolationMode,
)
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from collections.abc import Sequence
from PIL import Image


class CoverResize(torch.nn.Module):
    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias="warn",
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not isinstance(size, Sequence) or len(size) != 2:
            raise ValueError("size must be (h, w)")
        self.size = size
        self.max_size = max_size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, Image.Image):
            w, h = img.width, img.height
        else:
            h, w = img.shape[-2], img.shape[-1]

        scale_factor_h = self.size[0] / h
        scale_factor_w = self.size[1] / w

        scale_factor = (
            scale_factor_h if scale_factor_h > scale_factor_w else scale_factor_w
        )
        new_size = (round(h * scale_factor), round(w * scale_factor))

        return F.resize(
            img, new_size, self.interpolation, self.max_size, self.antialias
        )

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


def get_webvid_spatial_transform(
    resolution=[256, 512], resize_resolution=None, random_crop=False
):
    resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
    if resize_resolution is None:
        resize_resolution = min(resolution)
    if random_crop:
        spatial_transform = transforms.Compose(
            [
                transforms.Resize(resize_resolution),
                transforms.RandomCrop(resolution),
            ]
        )
    else:
        spatial_transform = transforms.Compose(
            [
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(resolution),
            ]
        )
    return spatial_transform


def get_video_spatial_transform(
    resolution=[256, 512], resize_resolution=None, random_crop=False
):
    resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
    if resize_resolution is None:
        resize_resolution = resolution
    if random_crop:
        spatial_transform = transforms.Compose(
            [
                CoverResize(resize_resolution),
                transforms.RandomCrop(resolution),
            ]
        )
    else:
        spatial_transform = transforms.Compose(
            [
                CoverResize(resize_resolution),
                transforms.CenterCrop(resolution),
            ]
        )
    return spatial_transform


def get_image_transform(resolution=[256, 512], to_tensor=True):
    resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
    image_transform = [
        transforms.Resize(min(resolution)),
        transforms.CenterCrop(resolution),
    ]
    if to_tensor:
        image_transform.append(transforms.ToTensor())
    spatial_transform = transforms.Compose(image_transform)
    return spatial_transform


if __name__ == "__main__":
    resize = CoverResize((576, 512))

    image = torch.randn(1, 3, 512, 512)
    out = resize(image)
    print(out.shape)
