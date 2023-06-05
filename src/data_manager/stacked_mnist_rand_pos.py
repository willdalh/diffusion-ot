import torch
import numpy as np
from torchvision.datasets import MNIST
import os

class StackedMNISTRandPos(MNIST):
    """
    Stacks three MNIST images on top of each other to form a new image.
    With random positions along the x-axis.
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

        imgs = [img, img2, img3]

        # Stack the images
        final_img = torch.ones(img.shape[0], img.shape[1] * 3, img.shape[2] * 3) * -1

        h, w = img.shape[1], img.shape[2]
        for i, im in enumerate(imgs):
            # Random position along x-axis
            x = np.random.randint(0, 3 * w - w)
            final_img[:, i * h : (i + 1) * h, x : x + w] = im

        return final_img, [target, target2, target3]
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "processed")