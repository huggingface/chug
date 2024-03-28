import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter


class AlignLongAxis:
    def __init__(
            self,
            input_size,
            interpolation=transforms.InterpolationMode.BICUBIC
    ):
        self.input_size = input_size
        self.interpolation = interpolation

    def __call__(self, img):
        img_width, img_height = F.get_image_size(img)
        if (
            (self.input_size[0] > self.input_size[1] and img_width > img_height) or
            (self.input_size[0] < self.input_size[1] and img_width < img_height)
        ):
            img = F.rotate(img, angle=-90, expand=True, interpolation=self.interpolation)
        return img


class Bitmap:
    def __init__(self, threshold=200):
        self.lut = [0 if i < threshold else i for i in range(256)]

    def __call__(self, img):
        if img.mode == "RGB" and len(self.lut) == 256:
            lut = self.lut + self.lut + self.lut
        else:
            lut = self.lut
        return img.point(lut)


class Erosion:
    def __init__(self, scale=3):
        super().__init__()
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    @staticmethod
    def get_params(scale):
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            scale = random.choice(scale)
        return scale

    def __call__(self, img):
        kernel_size = self.get_params(self.scale)
        if isinstance(img, torch.Tensor):
            padding = kernel_size // 2
            img = -torch.nn.functional.max_pool2d(-img, kernel_size=kernel_size, stride=1, padding=padding)  # minpool
        elif isinstance(img, Image.Image):
            img = img.filter(ImageFilter.MinFilter(kernel_size))
        return img


class Dilation:
    def __init__(self, scale=3):
        super().__init__()
        self.scale = scale

    @staticmethod
    def get_params(scale):
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            scale = random.choice(scale)
        return scale

    def __call__(self, img):
        kernel_size = self.get_params(self.scale)
        if isinstance(img, torch.Tensor):
            padding = kernel_size // 2
            img = torch.nn.functional.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=padding)
        elif isinstance(img, Image.Image):
            img = img.filter(ImageFilter.MaxFilter(kernel_size))
        return img


def python_find_non_zero(image: np.array):
    """This is a reimplementation of a findNonZero function equivalent to cv2."""
    non_zero_indices = np.column_stack(np.nonzero(image))
    idxvec = non_zero_indices[:, [1, 0]]
    idxvec = idxvec.reshape(-1, 1, 2)
    return idxvec


def python_bounding_rect(coordinates):
    """This is a reimplementation of a BoundingRect function equivalent to cv2."""
    min_values = np.min(coordinates, axis=(0, 1)).astype(int)
    max_values = np.max(coordinates, axis=(0, 1)).astype(int)
    x_min, y_min = min_values[0], min_values[1]
    width = max_values[0] - x_min + 1
    height = max_values[1] - y_min + 1
    return x_min, y_min, width, height


class CropMargin:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        image,
        gray_threshold: int = 200,
    ) -> np.array:
        # FIXME check tensor vs PIL and convert as needed, this is assuming PIL right now
        assert not isinstance(image, torch.Tensor)
        data = np.array(image.convert("L")).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = python_find_non_zero(gray)
        x_min, y_min, width, height = python_bounding_rect(coords)
        image = F.crop(image, y_min, x_min, height, width)
        return image


class ConvertColor:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        assert isinstance(image, Image.Image)
        return image.convert(self.mode)
