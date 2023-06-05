import torch
import numpy as np
from torchvision.datasets import MNIST
import os

class StackedMNIST(MNIST):
    """
    Stacks three MNIST images on top of each other to form a new image.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Two random indices
        index2 = np.random.randint(0, len(self))
        index3 = np.random.randint(0, len(self))

        # Get the images
        img2, target2 = super().__getitem__(index2)
        img3, target3 = super().__getitem__(index3)

        # Stack the images
        img = torch.cat([img, img2, img3], dim=1) # Height dimension
        return img, [target, target2, target3]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "processed")
