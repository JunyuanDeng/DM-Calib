import os
import random
from copy import deepcopy
from math import ceil, exp, log, log2, log10, tanh
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF


class RandomColorJitter:
    def __init__(self, level, prob=0.9):
        self.level = level
        self.prob = prob
        self.list_transform = [
            self._adjust_brightness_img,
            # self._adjust_sharpness_img,
            self._adjust_contrast_img,
            self._adjust_saturation_img,
            self._adjust_color_img,
        ]

    def _adjust_contrast_img(self, img, factor=1.0):
        """Adjust the image contrast."""
        return TF.adjust_contrast(img, factor)

    def _adjust_sharpness_img(self, img, factor=1.0):
        """Adjust the image contrast."""
        return TF.adjust_sharpness(img, factor)

    def _adjust_brightness_img(self, img, factor=1.0):
        """Adjust the brightness of image."""
        return TF.adjust_brightness(img, factor)

    def _adjust_saturation_img(self, img, factor=1.0):
        """Apply Color transformation to image."""
        return TF.adjust_saturation(img, factor / 2.0)

    def _adjust_color_img(self, img, factor=1.0):
        """Apply Color transformation to image."""
        return TF.adjust_hue(img, (factor - 1.0) / 4.0)

    def __call__(self, img):
        """Call function for color transformation.
        Args:
            img (dict): img dict from loading pipeline.

        Returns:
            dict: img after the transformation.
        """
        random.shuffle(self.list_transform)
        for op in self.list_transform:
            if np.random.random() < self.prob:
                factor = 1.0 + (
                    (self.level[1] - self.level[0]) * np.random.random() + self.level[0]
                )
                op(img, factor)
        return img


class RandomGrayscale:
    def __init__(self, prob=0.1, num_output_channels=3):
        super().__init__()
        self.prob = prob
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        if np.random.random() > self.prob:
            return img

        img = TF.rgb_to_grayscale(
            img, num_output_channels=self.num_output_channels
        )
        return img


class RandomGamma:
    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def __call__(self, img, level=None):
        if np.random.random() > self.prob:
            return img
        factor = (self.level[1] - self.level[0]) * np.random.rand() + self.level[0]

        img = TF.adjust_gamma(img, 1 + factor)
        return img


class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0), prob=0.9):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob
        self.padding = kernel_size // 2

    def apply(self, x, kernel):
        # Pad the input tensor
        x = F.pad(
            x.unsqueeze(0), (self.padding, self.padding, self.padding, self.padding), mode="reflect"
        )
        # Apply the convolution with the Gaussian kernel
        return F.conv2d(x, kernel, stride=1, padding=0, groups=x.size(1)).squeeze()

    def _create_kernel(self, sigma):
        # Create a 1D Gaussian kernel
        kernel_1d = torch.exp(
            -torch.arange(-self.padding, self.padding + 1) ** 2 / (2 * sigma**2)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Expand the kernel to 2D and match size of the input
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.view(1, 1, self.kernel_size, self.kernel_size).expand(
            3, 1, -1, -1
        )
        return kernel_2d

    def __call__(self, img):
        if np.random.random() > self.prob:
            return img
        sigma = (self.sigma[1] - self.sigma[0]) * np.random.rand() + self.sigma[0]
        kernel = self._create_kernel(sigma)

        img = self.apply(img, kernel)
        return img


augmentations_dict = {
    "Jitter": RandomColorJitter((-0.4, 0.4), prob=0.4),
    "Gamma": RandomGamma((-0.2, 0.2), prob=0.4),
    "Blur": GaussianBlur(kernel_size=13, sigma=(0.1, 2.0), prob=0.1),
    "Grayscale": RandomGrayscale(prob=0.1),
}
