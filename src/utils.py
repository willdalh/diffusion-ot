from collections import OrderedDict
import re
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid
import numpy as np
from typing import List, Tuple
from PIL import Image
import cv2
import math


# * FOR VISUALIZATION

def plot_images(images, title=None, clip=False, scale=1, nrow=-1, padding=2, pad_value=0):
    """
    Plot images in a grid using matplotlib
    
    Args:
        images (torch.Tensor): Tensor of shape (batch, channels, height, width)
        title (str, optional): Title of the plot. Defaults to None.
        clip (bool, optional): Clip the images to [0, 1]. Defaults to False.
        scale (float, optional): plot scale. Defaults to 1.
        nrow (int, optional): Number of images per row (-1 for all on 1 row). Defaults to -1.
        padding (int, optional): Padding between images. Defaults to 2.
        pad_value (float, optional): Padding value. Defaults to 0.
    """
    imgs = images.detach().clone()
    if clip:
        imgs = imgs.clamp(0, 1)
    if nrow == -1:
        nrow = imgs.shape[0]
    to_plot = make_grid(imgs, nrow=nrow, padding=padding, pad_value=pad_value).unsqueeze(0)

    # n = to_plot.shape[0]
    ax = plt.axes()
    fig = ax.get_figure()
    fig.set_size_inches(fig.get_size_inches() * scale)
    if title:
        fig.suptitle(title, fontsize=20)

    # axs = axs.flatten() if n > 1 else [axs]
    # for i, ax in enumerate(axs):
    img = tvf.to_pil_image(to_plot[0])
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax.axis("off")
    ax.matshow(np.asarray(img))
    plt.show()
           
def to_pils(x):
    """
    Turn batch of images into a list of PIL images

    Args:
        x (torch.Tensor): Tensor of shape (batch, channels, height, width)

    Returns:
        List[PIL.Image.Image]: List of PIL images
    """
    return [tvf.to_pil_image(img) for img in x]

def make_grid_from_temporal_batch(x, nrow=-1, padding=0, pad_value=0):
    """
    Make a temporal grid of images from a temporal batch of images

    Args:
        x (torch.Tensor): Tensor of shape (time, batch, channels, height, width)
        nrow (int, optional): Number of images per row (-1 for all on 1 row). Defaults to -1.
        padding (int, optional): Padding between images. Defaults to 0.
        pad_value (float, optional): Padding value. Defaults to 0.

    Returns:
        torch.Tensor: Tensor of shape (time, channels, new_height, new_width)
    """
    if nrow == -1:
        nrow = x.shape[1]
    time_grids = [make_grid(e, nrow=nrow, padding=padding, pad_value=pad_value) for e in x]
    grid = torch.stack(time_grids)
    return grid

def scale_images(ims: List[Image.Image], scale: int):
    """
    Scale a list of images

    Args:
        ims (List[PIL.Image.Image]): List of images
        scale (int): Scale factor

    Returns:
        List[PIL.Image.Image]: List of scaled images
    """
    new_ims = []
    for im in ims: 
        orig_w, orig_h = im.size
        new_im = im.resize((scale * orig_w, scale * orig_h), resample=Image.BOX)
        new_ims.append(new_im)
    return new_ims

def save_gif(data: torch.Tensor, path: str, fps=30, scale=1):
    """
    Save a tensor of images as a gif

    Args:
        data (torch.Tensor): Tensor of shape (time, channels, height, width)
        path (str): Path to save gif to
        fps (int, optional): Frames per second. Defaults to 30.
        scale (int, optional): Scale factor. Defaults to 1.
    """
    assert len(data.shape) == 4, "Data must be a 4D tensor (batch, channels, height, width)"
    assert path.endswith(".gif"), "Path must end with .gif"

    imgs: List[Image.Image] = to_pils(data)
    imgs = scale_images(imgs, scale)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000/fps, loop=0)

def save_video(data: torch.Tensor, path: str, fps=30, scale=1):
    """
    Save a tensor of images as a video

    Args:
        data (torch.Tensor): Tensor of shape (time, channels, height, width)
        path (str): Path to save video to with .mp4 extension
        fps (int, optional): Frames per second. Defaults to 30.
        scale (int, optional): Scale factor. Defaults to 1.
    """
    assert len(data.shape) == 4, "Data must be a 4D tensor (batch, channels, height, width)"
    assert path.endswith(".mp4"), "Path must end with .mp4"

    imgs: List[Image.Image] = to_pils(data)
    imgs = scale_images(imgs, scale)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (imgs[0].width, imgs[0].height), data.shape[1] == 3)
    for img in imgs:
        if data.shape[1] == 1:
            out.write(np.asarray(img))
        else:
            out.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    out.release()


def normalize(x, across_batch=False):
    """
    Normalize a tensor of images
    
    Args:
        x (torch.Tensor): Tensor of shape (batch, channels, height, width)
        across_batch (bool, optional): Whether to normalize across the batch. Defaults to False.

    Returns:
        torch.Tensor: Normalized tensor
    """
    x_copy = x.detach().clone()
    out = None
    if across_batch:
        out = _normalize_across_all_dims(x_copy)
    else:
        for i in range(x_copy.shape[0]):
            x_copy[i] = _normalize_across_all_dims(x_copy[i])
        out = x_copy
    return out
        

def _normalize_across_all_dims(x):
    """
    Normalize a tensor of images across all dimensions

    Args:
        x (torch.Tensor): Tensor of shape (channels, height, width)

    Returns:
        torch.Tensor: Normalized tensor
    """
    inv_scale = x.max() - x.min()
    if inv_scale == 0:
        inv_scale = 1
    return (x - x.min()) / (inv_scale)


# * Vector space things

def slerp(low, high, val):
    """
    Spherical interpolation between two points
    Function definition from https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355 

    Args:
        low (torch.Tensor): Tensor of shape (batch_size, *)
        high (torch.Tensor): Tensor of shape (batch_size, *)
        val (float): weight

    Returns:
        Tensor of shape (batch_size, *)
    """
    low_norm = low/torch.linalg.norm(low)
    high_norm = high/torch.linalg.norm(high)
    dot = (low_norm*high_norm).sum()
    if torch.abs(dot) > 0.9995:
        return low + val * (high - low) # if the inputs are too close, linearly interpolate

    omega = torch.acos(dot)
    omega_val = omega * val
    sin_omega = torch.sin(omega)
    sin_omega_val = torch.sin(omega_val)

    s1 = torch.sin(omega - omega_val) / sin_omega
    s2 = sin_omega_val / sin_omega
    res = s1 * low + s2 * high
    return res

def spherical_interpolation(start: torch.Tensor, end: torch.Tensor, n=2):
    """
    Interpolates two tensors using slerp

    Args:
        start (torch.Tensor): Tensor of shape (batch_size, *)
        end (torch.Tensor): Tensor of shape (batch_size, *)
        n (int): number of final points (n=3 gives 1 interpolated value)

    Returns:
        Tensor of shape (n, *)
    """
    assert n >= 2, "n must be >= 2"
    res = torch.cat([slerp(start, end, i/(n-1)) for i in range(n)], dim=0)
    return res

def linear_interpolation(start: torch.Tensor, end: torch.Tensor, n=2):
    """
    Interpolates two tensors linearly

    Args:
        start (torch.Tensor): Tensor of shape (batch_size, *)
        end (torch.Tensor): Tensor of shape (batch_size, *)
        n (int): number of final points (n=3 gives 1 interpolated value)
    
    Returns:
        Tensor of shape (n, *)
    """
    assert n >= 2, "n must be >= 2"
    res = torch.cat([torch.lerp(start, end, i/(n-1)) for i in range(n)], dim=0)
    return res

# * Probability things
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

def handle_mus_and_sigmas(mus, sigmas):
    """
    Helper function to format mus and sigmas correctly

    Args:
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations

    Returns:
        Tuple of (mus, sigmas)
    """
    if type(mus) == float or type(mus) == int: # Wrap to list
        mus = [mus]
        sigmas = [sigmas]
    if type(mus[0]) == int or type(mus[0]) == float: # Wrap to n-dimensional case
        mus = [[m] for m in mus]
        sigmas = [[[s]] for s in sigmas]
    return mus, sigmas

def gaussian_pdf_at(x, mus, sigmas):
    """
    Computes the probability density function of a gaussian at a point

    Args:
        x (torch.Tensor | float | list): Point(s) to evaluate at
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations

    Returns:
        Probability density function of a gaussian at x
    """
    mus, sigmas = handle_mus_and_sigmas(mus, sigmas)
    
    dim = len(mus[0])
    if dim == 1:
        return sum([norm.pdf(x, loc=mus[i][0], scale=sigmas[i][0][0]) for i in range(len(mus))])/len(mus)
    elif dim == 2:
        return sum([mvn.pdf(x, mean=mus[i], cov=sigmas[i]) for i in range(len(mus))])/len(mus)
    
def gaussian_cdf_at(x, mus, sigmas):
    """
    Computes the cumulative density function of a gaussian at a point
    
    Args:
        x (torch.Tensor | float | list): Point(s) to evaluate at
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations

    Returns:
        Cumulative density function of a gaussian at x

    """
    mus, sigmas = handle_mus_and_sigmas(mus, sigmas)
    
    dim = len(mus[0])
    if dim == 1:
        return sum([norm.cdf(x, loc=mus[i][0], scale=sigmas[i][0][0]) for i in range(len(mus))])/len(mus)
    elif dim == 2:
        return sum([mvn.cdf(x, mean=mus[i], cov=sigmas[i]) for i in range(len(mus))])/len(mus)
    
def gaussian_ppf_at(x, mus, sigmas):
    """
    Computes the percent point function of a gaussian at a point
    
    Args:
        x (torch.Tensor | float | list): Point(s) to evaluate at [0, 1]
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations

    Returns:
        Percent point function of a gaussian at x
     
    """
    mus, sigmas = handle_mus_and_sigmas(mus, sigmas)
    
    dim = len(mus[0])
    if dim == 1:
        return sum([norm.ppf(x, loc=mus[i][0], scale=sigmas[i][0][0]) for i in range(len(mus))])*len(mus)
    elif dim == 2:
        return sum([mvn.ppf(x, mean=mus[i], cov=sigmas[i]) for i in range(len(mus))])*len(mus)

def get_gaussian_pdf(mus, sigmas, start=-2, end=2, n=1000):
    """
    Computes the probability density function of a gaussian
    
    Args:
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations
        start (float | list): Start of the range to evaluate at
        end (float | list): End of the range to evaluate at
        n (int): Number of points to evaluate at

    Returns:
        Tuple of (x, y) where x is the range and y is the pdf
    """

    mus, sigmas = handle_mus_and_sigmas(mus, sigmas)
    dim = len(mus[0])
    assert dim in [1, 2], "Only 1D and 2D distributions are supported"

    if dim == 1:
        x = np.linspace(start, end, n)
        y = np.zeros_like(x)
        for mu, sigma in zip(mus, sigmas):
            y += norm.pdf(x, loc=mu[0], scale=sigma[0][0]) / len(mus)
        return x, y
    
    elif dim == 2:
        if type(start) == int or type(start) == float:
            start = [start, start]
            end = [end, end]
        sigmas = np.array(sigmas)
        # sigmas = np.matmul(sigmas, sigmas.transpose(0, 1, 2))
        x = np.linspace(start[0], end[0], n)
        y = np.linspace(start[1], end[1], n)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Z = np.zeros_like(X)
        for mu, cov in zip(mus, sigmas):
            Z += mvn.pdf(pos, mean=mu, cov=cov) / len(mus)
        return pos, Z

def get_gaussian_cdf(mus, sigmas, start=-2, end=2, n=1000):
    """
    Computes the cumulative density function of a gaussian

    Args:
        mus (list): List of means
        sigmas (list): List of covariances/standard deviations
        start (float | list): Start of the range to evaluate at
        end (float | list): End of the range to evaluate at
        n (int): Number of points to evaluate at
        
    Returns:
        Tuple of (x, y) where x is the range and y is the cdf
    """

    mus, sigmas = handle_mus_and_sigmas(mus, sigmas)
    dim = len(mus[0])
    assert dim in [1, 2], "Only 1D and 2D distributions are supported"

    if dim == 1:
        x = np.linspace(start, end, n)
        y = np.zeros_like(x)
        for mu, sigma in zip(mus, sigmas):
            y += norm.cdf(x, loc=mu[0], scale=sigma[0][0]) / len(mus)
        return x, y
    
    elif dim == 2:
        if type(start) == int or type(start) == float:
            start = [start, start]
            end = [end, end]
        sigmas = np.array(sigmas)
        # sigmas = np.matmul(sigmas, sigmas.transpose(0, 1, 2))
        x = np.linspace(start[0], end[0], n)
        y = np.linspace(start[1], end[1], n)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Z = np.zeros_like(X)
        for mu, cov in zip(mus, sigmas):
            Z += mvn.cdf(pos, mean=mu, cov=cov) / len(mus)
        return pos, Z



# * Diffusion things
def get_schedule(name: str, betas: Tuple[float, float], n_T: int):
    """
    Returns a schedule for the beta values

    Args:
        name (str): Name of the schedule
        betas (Tuple[float, float]): Tuple of (beta_1, beta_T)
        n_T (int): Number of time steps

    Returns:
        Schedule for the beta values
    """

    if name == "linear":
        return linear_schedule(betas, n_T)
    elif name == "scaled_linear":
        return scaled_linear_schedule(betas, n_T)
    elif name == "cosine":
        return cosine_schedule(n_T)
    else:
        raise ValueError(f"Unknown schedule {name}")

def linear_schedule(betas: Tuple[float, float], n_T: int):
    """
    Linear beta schedule

    Args:
        betas (Tuple[float, float]): Tuple of (beta_1, beta_T)
        n_T (int): Number of time steps

    Returns:
        Schedule for the beta values
    """
    beta = torch.linspace(betas[0], betas[1], n_T + 1)
    sqrt_beta = torch.sqrt(beta)
    
    alpha = 1 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    sqrt_alphabar = torch.sqrt(alphabar)
    sqrt_m_alphabar = torch.sqrt(1 - alphabar)
    oneover_sqrt_alpha = 1 / torch.sqrt(alpha)
    malpha_over_sqrtmab = (1 - alpha) / sqrt_m_alphabar

    return {
        "beta": beta,
        "sqrt_beta": sqrt_beta,
        "alpha": alpha,
        "alphabar": alphabar,
        "sqrt_alphabar": sqrt_alphabar,
        "sqrt_m_alphabar": sqrt_m_alphabar,
        "oneover_sqrt_alpha": oneover_sqrt_alpha,
        "malpha_over_sqrtmab": malpha_over_sqrtmab
    }

def scaled_linear_schedule(betas: Tuple[float, float], n_T: int):
    """
    Scaled linear beta schedule

    Args:
        betas (Tuple[float, float]): Tuple of (beta_1, beta_T)
        n_T (int): Number of time steps
        
    Returns:
        Schedule for the beta values
    """
    beta = torch.linspace(betas[0]**0.5, betas[1]**0.5, n_T + 1)**2
    sqrt_beta = torch.sqrt(beta)
    
    alpha = 1 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    sqrt_alphabar = torch.sqrt(alphabar)
    sqrt_m_alphabar = torch.sqrt(1 - alphabar)
    oneover_sqrt_alpha = 1 / torch.sqrt(alpha)
    malpha_over_sqrtmab = (1 - alpha) / sqrt_m_alphabar

    return {
        "beta": beta,
        "sqrt_beta": sqrt_beta,
        "alpha": alpha,
        "alphabar": alphabar,
        "sqrt_alphabar": sqrt_alphabar,
        "sqrt_m_alphabar": sqrt_m_alphabar,
        "oneover_sqrt_alpha": oneover_sqrt_alpha,
        "malpha_over_sqrtmab": malpha_over_sqrtmab
    }

def cosine_schedule(n_T):
    """
    Cosine beta schedule

    Args:
        n_T (int): Number of time steps

    Returns:
        Schedule for the beta values
    """

    def f(t):
        return math.cos(((t/(n_T+1) + 0.008)/(1.008)) * math.pi/2)**2

    beta = torch.Tensor([1-f(t+1)/f(t) for t in range(n_T+1)])
    beta = beta.clip(0, 0.999)
    alpha = 1 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    
    sqrt_beta = torch.sqrt(beta)
    sqrt_alphabar = torch.sqrt(alphabar)
    sqrt_m_alphabar = torch.sqrt(1 - alphabar)
    oneover_sqrt_alpha = 1 /  torch.sqrt(alpha)
    malpha_over_sqrtmab = (1 - alpha) / sqrt_m_alphabar

    return {
        "beta": beta,
        "sqrt_beta": sqrt_beta,
        "alpha": alpha,
        "alphabar": alphabar,
        "sqrt_alphabar": sqrt_alphabar,
        "sqrt_m_alphabar": sqrt_m_alphabar,
        "oneover_sqrt_alpha": oneover_sqrt_alpha,
        "malpha_over_sqrtmab": malpha_over_sqrtmab
    }


# * Tensor things

def extract_tensor(singledim_tensor, ts, target_shape):
    """
    Extracts a tensor from a 1D tensor to a tensor with the same number of dimensions as elements in target_shape
    """
    res = singledim_tensor.to(ts.device)[ts]
    while res.ndim < len(target_shape):
        res = res[..., None]
    return res


# * Compatibility things

def fix_legacy_state_dict(state_dict):
    """
    Fix some compatibility issues with keys in old models
    Basically, old models had keys like "up1.conv.0.double_conv.1.weight"
    But after changing the model to dynamically construct the layers, the keys are now like "up.1.conv.0.double_conv.0.weight"
    See the extra "." after "up"

    Args:
        state_dict (OrderedDict): State dict to fix

    Returns:
        OrderedDict: Fixed state dict
    """
    
    new_list = []
    for key in state_dict.keys():
        first_num_index = re.search(r"\d", key)
        if first_num_index is not None:
            first_num_index = first_num_index.start()
            splitted = key.split(".")

            if splitted[0][-1] == key[first_num_index]:
                new_key = key[:first_num_index] + "." + f"{int(key[first_num_index])-1}" + "." + ".".join(splitted[1:])
                new_list.append((new_key, state_dict[key]))
            else:
                new_list.append((key, state_dict[key]))
        else:
            new_list.append((key, state_dict[key]))
    fixed_state_dict = OrderedDict(new_list)
    return fixed_state_dict

