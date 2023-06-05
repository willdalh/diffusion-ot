import torch
import numpy as np
from torchvision.datasets import MNIST
import os

class ColoredMNIST(MNIST):
    """
    Colored MNIST dataset, where each image is colored with a random color.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        self.max_val = super().__getitem__(0)[0].max()
        self.min_val = super().__getitem__(0)[0].min()
        self.diff = self.max_val - self.min_val

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        rand_color = self.random_color()
        
        img = torch.cat([(((img - self.min_val) / self.diff) * col) * self.diff + self.min_val for col in rand_color], dim=0) # Scale back to [0, 1] and multiply by color then scale back
        return img, target

    def random_color(self):
        return self.colors[np.random.randint(0, len(self.colors))]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "processed")