from utils import extract_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

# Blocks are inspired by https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py

class DoubleConv(nn.Module):
    """
    Double convolution block with residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """ 
    Downscaling with maxpool then double conv and learnable timestep embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.timestep_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        # print("In forward for down", x.shape)
        res = self.maxpool_conv(x)
        # print("In forward for down", x.shape)
        t_emb = self.timestep_emb(t)[:, :, None, None].repeat(1, 1, res.shape[-2], res.shape[-1])
        return res + t_emb

class Up(nn.Module):
    """
    Upscaling then double conv and learnable timestep embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsample + Conv instead of ConvTranspose: See 'Better Upsampling' at https://distill.pub/2016/deconv-checkerboard/
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.timestep_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        res = self.up(x)
        res = torch.cat([skip_x, res], dim=1)
        res = self.conv(res)
        t_emb = self.timestep_emb(t)[:, :, None, None].repeat(1, 1, res.shape[-2], res.shape[-1])
        return res + t_emb
    
class SelfAttention(nn.Module):
    """
    Self attention block
    Not used in final models
    """
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_val, _ = self.mha(x_ln, x_ln, x_ln)
        attention_val += x
        attention_val = self.ff_self(attention_val) + attention_val
        return attention_val.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)


