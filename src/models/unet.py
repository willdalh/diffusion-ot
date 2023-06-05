from collections import OrderedDict
import re
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn

from models.blocks import DoubleConv, Down, Up, SelfAttention
from models.model_utils import timestep_encoding
from typing import List

class UNet(nn.Module):
    """
    UNet architecture for the noise-predictor
    Inspired by https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, data_shape = (1, 32, 32), time_dim: int = 256, device="cuda" if torch.cuda.is_available() else "cpu", unet_start_channels: int = 64, unet_down_factors: List[int] = [2, 4, 4], unet_bot_factors: List[int] = [8, 8, 4], unet_use_attention: bool = True):
        super(UNet, self).__init__()
        # self.device = device
        self.to(device)
        self.time_dim = time_dim


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_start_channels = unet_start_channels
        self.unet_down_factors = unet_down_factors
        self.unet_bot_factors = unet_bot_factors
        self.unet_use_attention = unet_use_attention

        h, w = data_shape[1:]
        if unet_use_attention:
            # Assumes that the image is square
            assert h % 8 == 0, f"Width and height must be divisible by 8, got {h} which divides by 8 is {h / 8}"
            assert h == w, f"Width and height must be equal, got {h} and {w}"


        self.inc, self.down, self.bot, self.up, self.outc = self.construct_unet()
        # self.rename_layers()

        self.data_shape = data_shape
        self.to(device)
        # print(self)
        with torch.no_grad():
            self.forward(torch.randn(1, *self.data_shape).to(device), torch.LongTensor([6]).to(device))

        print("Finished testing UNet")

    def forward(self, indata, t):
        t = t.unsqueeze(-1).type(torch.float32)
        t = timestep_encoding(t, self.time_dim)

        x1 = self.inc(indata)

        res_xs = [x1]
        for i, layer in enumerate(self.down):
            x = layer(res_xs[-1], t)
            res_xs.append(x) if i != len(self.down) - 1 else None

        for i, layer in enumerate(self.bot):
            x = layer(x)

        res_xs = res_xs[::-1]

        for i, layer in enumerate(self.up):
            x = layer(x, res_xs[i], t)


        out = self.outc(x)
        return out

    def construct_unet(self):
        inc = DoubleConv(self.in_channels, self.unet_start_channels)

        # Downsampling
        down_layers = []
        for i, factor in enumerate(self.unet_down_factors):
            if i == 0:
                layer = Down(self.unet_start_channels, self.unet_start_channels * factor)
                # print(self.unet_start_channels, self.unet_start_channels * factor)
            else:
                layer = Down(self.unet_start_channels * self.unet_down_factors[i - 1], self.unet_start_channels * factor)
                # print(self.unet_start_channels * self.unet_down_factors[i - 1], self.unet_start_channels * factor)

            down_layers.append(layer)
        down = nn.ModuleList(down_layers)

        # Bottleneck
        bot_layers = []
        for i, factor in enumerate(self.unet_bot_factors):
            if i == 0:
                layer = DoubleConv(self.unet_start_channels * self.unet_down_factors[-1], self.unet_start_channels * factor)
            else:
                layer = DoubleConv(self.unet_start_channels * self.unet_bot_factors[i - 1], self.unet_start_channels * factor)
            bot_layers.append(layer)
        bot = nn.ModuleList(bot_layers)
        

        # The upsampling layers has skip connections
        up_layers = []
        down_factors_reversed = self.unet_down_factors[::-1]
        up_factors = [*down_factors_reversed, 1]

        # Upsamling
        # Get the channels from the downsampling layers at the same level
        for i, (factor, middle_factor, next_factor) in enumerate(zip(up_factors, up_factors[1:] + [np.nan], up_factors[2:] + [np.nan, np.nan])):

            if i == 0:
                layer = Up(self.unet_start_channels * (self.unet_bot_factors[-1] + middle_factor), int(self.unet_start_channels * next_factor))
            elif i == len(up_factors) - 2:
                layer = Up(self.unet_start_channels * factor, self.unet_start_channels * middle_factor)

            else:
                layer = Up(self.unet_start_channels * factor, int(self.unet_start_channels * next_factor))
                      
            up_layers.append(layer)
            if i == len(up_factors) - 2:
                break
        up = nn.ModuleList(up_layers)
        

        outc = nn.Conv2d(self.unet_start_channels, self.out_channels, kernel_size=1)

        return inc, down, bot, up, outc
        


    def get_model_stored_size(self):
        """
        Returns the size of the model in MB
        https://discuss.pytorch.org/t/finding-model-size/130275
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buf in self.buffers():
            buffer_size += buf.nelement() * buf.element_size()
        size = (param_size + buffer_size) / 1024 ** 2
        return size
