from typing import Dict, Tuple
from abc import ABC, abstractmethod

import json


from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from argparse import Namespace

from utils import  get_schedule, extract_tensor, fix_legacy_state_dict

from models.unet import UNet
from models.single_dim_net import SingleDimNet

class Denoiser(nn.Module, ABC):
    """
    Denoiser class that wraps the model and the diffusion process.
    """
    def __init__(self, model: nn.Module, betas: Tuple[float, float] = (1e-4, 0.02), n_T: int = 1000, criterion: nn.Module = nn.MSELoss(), schedule_name: str = "linear"):
        """
        Args:
            model (nn.Module): noise-predictor $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(x_t, t)$
            betas (Tuple[float, float]): Tuple of $\beta_1$ and $\beta_T$ for the DDPM framework
            n_T (int): Number of diffusion steps
            criterion (nn.Module): Loss function
            schedule_name (str): Name of the scheduler to use. DDPM framework uses a linear schedule, Stable Diffusion v1.5 uses quadratic schedule
        """
        super(Denoiser, self).__init__()
        self.model = model
        self.betas = betas
        self.n_T = n_T
        self.criterion = criterion
        self.data_shape = model.data_shape if model is not None else None
        
        schedule = get_schedule(schedule_name, betas, n_T)
        self._load_schedule(schedule)
        print("Denoiser initialized")



    def forward_process(self, x: torch.Tensor, ts=None, return_ts=False) -> torch.Tensor:
        """
        Forward process for generating noised instances from pure x_0
        Args:
            x (torch.Tensor): Pure instances
            ts (torch.Tensor): Time steps at which to corrupt instances
            return_ts (bool): Whether to return the time steps
        Returns:
            x_t (torch.Tensor): Corrupted instances
            eps (torch.Tensor): Noise
            ts (torch.Tensor): Time steps at which corrupt instances (if return_ts is True)
        """
        if ts is None:
            ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(x.device)
            # print(ts)
        eps = torch.randn_like(x) # Sample from N(0, 1)
        x_t = extract_tensor(self.sqrt_alphabar, ts, x.shape) * x + extract_tensor(self.sqrt_m_alphabar, ts, x.shape) * eps
        if return_ts:
            return x_t, eps, ts
        return x_t, eps

    def forward_process_loss(self, x: torch.Tensor, ts: torch.Tensor = None) -> torch.Tensor:
        """
        Corrupt instance and have the noise-predictor predict the noise, then compute the loss
        Args:
            x (torch.Tensor): Pure instances
            ts (torch.Tensor): Time steps at which to corrupt instances
        Returns:
            loss (torch.Tensor): Loss (using self.criterion)
        """
        x_t, eps, ts = self.forward_process(x, ts, return_ts=True)
        pred_eps = self.model(x_t, ts)
        loss = self.criterion(eps, pred_eps)
        return loss
    
    def _load_schedule(self, schedule: Dict[str, torch.Tensor]):
        """
        Load the schedule tensors onto the denoiser
        Args:
            schedule (Dict[str, torch.Tensor]): Dictionary of tensors to be loaded into the model
        """
        for k, v in schedule.items():
            self.register_buffer(k, v)
    
    def sample(self, device, n=1, latents=None):
        """
        Sample from the model
        Args:
            device (torch.device): Device to sample on
            n (int, optional): Number of samples to generate
            latents (torch.Tensor, optional): Latents x_T to start the sampling process from. Overrides n
        Returns:
            torch.Tensor: x_0
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.randn(n, *self.data_shape).to(device) if latents is None else latents.clone().detach().to(device)
            for i in range(self.n_T, 0, -1):
                x_t = self.sample_step(x_t, i)
        return x_t

    @abstractmethod
    def sample_step(self, x, t) -> torch.Tensor:
        """
        Sample from the model at a particular time step. Ov
        Args:
            x (torch.Tensor): x_t 
            t (int): Time step
        Returns:
            torch.Tensor: x_{t-1}
        """
        pass

    def save_model(self, path: str):
       
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str, device):
        """
        Load the model from a path
        Args:
            path (str): Path to the model
            device (torch.device): Device to load the model onto
        """
        self.model.load_state_dict(torch.load(path, map_location=device))

    def get_predicted_x0(self, x, t, eps_pred=None):
        tt = None
        if isinstance(t, int):
            tt = torch.LongTensor([t] * x.shape[0]).to(x.device)
        if isinstance(t, list):
            tt = torch.LongTensor(t).to(x.device)
        if isinstance(t, torch.LongTensor):
            assert len(t.shape) == 1
            tt = t
        tt = tt.to(x.device)
        # x0_t = (x - self.sqrt_m_alphabar[tt, None, None, None] * self.model(x, tt)) / self.sqrt_alphabar[tt, None, None, None]
        eps = self.model(x, tt) if eps_pred is None else eps_pred
        x0_t = (x - extract_tensor(self.sqrt_m_alphabar, tt, x.shape) * eps) / extract_tensor(self.sqrt_alphabar, tt, x.shape)
        return x0_t
    
    @classmethod
    def load_from_training_log(cls, log_dir, model_name, device):
        """
        Load the denoiser from a training log
        Args:
            log_dir (str): Path to the training log
            model_name (str): Name of the model to load
            device (torch.device): Device to load the model onto
        Returns:
            Denoiser: Denoiser loaded from the training log
        """
        with open(f"{log_dir}/args.json", "r") as f:
            training_args = json.load(f)
            training_args = Namespace(**training_args)
            training_args.data_shape = tuple(training_args.data_shape)

        print("Loading model...")
        model_type = training_args.model_type
        if model_type == "unet":
            assert len(training_args.data_shape) == 3
            c, h, w = training_args.data_shape
            model = UNet(in_channels=c, out_channels=c, data_shape=training_args.data_shape, time_dim=256, unet_start_channels=training_args.unet_start_channels, unet_down_factors=training_args.unet_down_factors, unet_bot_factors=training_args.unet_bot_factors, unet_use_attention=training_args.unet_use_attention)
            try:
                model.load_state_dict(torch.load(f"{log_dir}/models/{model_name}", map_location=device))
            except:
                state_dict = torch.load(f"{log_dir}/models/{model_name}", map_location=device)
                print("Detected potential compatibility issues with model. Attempting to fix...")
                loaded = fix_legacy_state_dict(state_dict)
                model.load_state_dict(loaded)
                print("Fixed 👍")
                
        elif model_type == "single_dim_net":
            model = SingleDimNet(in_features=training_args.data_shape[0], out_features=training_args.data_shape[0], data_shape=training_args.data_shape)
            model.load_state_dict(torch.load(f"{log_dir}/models/{model_name}", map_location=device))

        # Create instance of class
        print("Creating instance of class...")
        if cls.__name__ == "DDPM":
            denoiser = DDPM(model, betas=(training_args.beta1, training_args.beta2), n_T=training_args.n_T, schedule_name=training_args.scheduler)
        elif cls.__name__ == "DDIM":
            denoiser = DDIM(model, betas=(training_args.beta1, training_args.beta2), n_T=training_args.n_T, schedule_name=training_args.scheduler, eta=0)

        denoiser.to(device)
        return denoiser
            



class DDPM(Denoiser):
    """
    Denoiser class for DDPM
    """
    def __init__(self, model: UNet, betas: Tuple[float, float] = (1e-4, 0.02), n_T: int = 1000, criterion: nn.Module = nn.MSELoss(), schedule_name: str = "linear"):
        """
        Args:
            model (UNet): noise-predictor
            betas (Tuple[float, float], optional): Start and endpoints for the variance schedule. Defaults to (1e-4, 0.02).
            n_T (int, optional): Number of time steps. Defaults to 1000.
            criterion (nn.Module, optional): Loss function. Defaults to nn.MSELoss().
            schedule_name (str, optional): Name of the schedule. Defaults to "linear".
        """

            
        super(DDPM, self).__init__(model, betas, n_T, criterion, schedule_name)

    def sample_step(self, x, t):
        """
        Perform one denoising step

        Args:
            x (torch.Tensor): input latent
            t (int): Time step

        Returns:
            torch.Tensor: less noised latent
        """

        z = torch.randn_like(x).to(x.device) if t > 1 else 0 
        tt = torch.Tensor([t] * x.shape[0]).to(x.device) # Tensor of shape (N,)
        eps = self.model(x, tt)
        x_tminusone = self.oneover_sqrt_alpha[t] * (x - eps * self.malpha_over_sqrtmab[t]) + self.sqrt_beta[t] * z
        return x_tminusone


class DDIM(Denoiser):
    """
    Denoiser class for DDIM
    """
    def __init__(self, model: UNet, betas: Tuple[float, float] = (1e-4, 0.02), n_T: int = 1000, criterion: nn.Module = nn.MSELoss(), schedule_name: str = "linear", eta: float = 0):

        """
        Args:
            model (UNet): noise-predictor
            betas (Tuple[float, float], optional): Start and endpoints for the variance schedule. Defaults to (1e-4, 0.02).
            n_T (int, optional): Number of time steps. Defaults to 1000.
            criterion (nn.Module, optional): Loss function. Defaults to nn.MSELoss().
            schedule_name (str, optional): Name of the schedule. Defaults to "linear".
            eta (float, optional): Parameter controlling the stochasticity of the sampling. 0=deterministic, 1=stochastic. Defaults to 0.

        """
        super(DDIM, self).__init__(model, betas, n_T, criterion, schedule_name)
        self.eta = eta

        self.timesteps = None
        self.num_inference_timesteps = None


        if len(model.data_shape) == 3:
            self.clip_range = 1
        elif len(model.data_shape) == 1:
            self.clip_range = 50


    def set_inference_timesteps(self, num_inference_timesteps):
        """
        Set the number of inference timesteps
        """
        ratio = (self.n_T) // num_inference_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        timesteps = (np.arange(0, num_inference_timesteps) * ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def sample_step(self, x, t, last_step=False, eps_pred=None):
        """
        Perform one denoising step 

        Args:
            x (torch.Tensor): input latent
            t (int): Time step
            last_step (bool, optional): Whether this is the last step. Defaults to False.
            eps_pred (torch.Tensor, optional): Predicted noise. Defaults to None.
        
        Returns:
            torch.Tensor: less noised latent
        """
        prev_t = t - self.n_T // self.num_inference_timesteps
        alphabar_t = self.alphabar[t]
        alphabar_t_prev = self.alphabar[prev_t]
        if prev_t < 0:
            alphabar_t_prev = torch.tensor(1.0).to(x.device)

        tt = torch.Tensor([t] * x.shape[0]).to(x.device)
        z = torch.randn_like(x).to(x.device) if not last_step else 0
        eps = self.model(x, tt) if eps_pred is None else eps_pred 

        x0_t = (x - eps * (1 - alphabar_t).sqrt()) / alphabar_t.sqrt() # Predicted x0
        x0_t = x0_t.clip(-self.clip_range, self.clip_range)
        
        c1 = self.eta * ((1 - alphabar_t / alphabar_t_prev) * (1 - alphabar_t_prev) / (
                1 - alphabar_t)).sqrt() # Scaling for the noise added
        c2 = ((1 - alphabar_t_prev) - c1 ** 2).sqrt() # Scaling for the predicted noise
        # if last_step:
        #     c1 = 0
        x = alphabar_t_prev.sqrt() * x0_t + c1 * z + c2 * eps
        return x
    
    def sample(self, device, timesteps=None, n=1, latents=None, get_trajectory=False):
        """
        Sample from the model

        Args:
            device (torch.device): Device to sample on
            timesteps (int, optional): Number of timesteps. Defaults to n_T
            n (int, optional): Number of samples. Defaults to 1.
            latents (torch.Tensor, optional): Latents to start from. Defaults to None.
            get_trajectory (bool, optional): Whether to return the full trajectory. Defaults to False.

        Returns:
            torch.Tensor: Sampled data
            torch.Tensor: Trajectory of sampled latents if get_trajectory=True
        """

        if timesteps is None and self.timesteps is None:
            raise ValueError("Must set timesteps as ddim.timesteps is None")
        if timesteps is not None:
            self.set_inference_timesteps(timesteps)
    

        with torch.no_grad():
            x_i = torch.randn(n, *self.data_shape).to(device) if latents is None else latents.clone().detach().to(device)
            if get_trajectory:
                xs = [x_i] # Save trajectory
            for i in self.timesteps:
                last_step = i == self.timesteps[-1]
                x_i = self.sample_step(x_i, i, last_step)
                if get_trajectory:
                    xs.append(x_i)

        if get_trajectory:
            return x_i, torch.stack(xs)
        return x_i
