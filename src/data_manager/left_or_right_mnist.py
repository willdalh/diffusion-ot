import torch
import numpy as np
from torchvision.datasets import MNIST
import os

class LeftOrRightMNIST(MNIST):
    """
    Places an MNIST digit on either the left or right side of the image based on the position argument.
    position: 0 for left, 1 for right
    """
    def __init__(self, position, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = position

    def __getitem__(self, index):
        img, target = super().__getitem__(index) # Will already be transformed to tensor and normalized here
        final_img = torch.ones(img.shape[0], img.shape[1], img.shape[2] * 2) * -1
        if self.position == 0:
            final_img[:, :, :img.shape[2]] = img
        elif self.position == 1:
            final_img[:, :, img.shape[2]:] = img
        
        return final_img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "processed")
