
import sys
import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn

from denoiser import DDPM
from models.unet import UNet
from models.single_dim_net import SingleDimNet
from data_manager.get_dataset import get_dataset
from trainer import Trainer

from custom_types import TrainingArgsType

import argparse
import shutil
import os
import json

def str_to_bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def run_training(args: TrainingArgsType):
    RUNNING_ON_SLURM = "SLURM_JOB_ID" in os.environ
    print(f"Running on SLURM: {RUNNING_ON_SLURM}")

    args.log_dir = "logs/training/" + args.log_name # * Additional arg

    # ! PROTECTED DIRS
    protected_dirs = ["Celeb256", "Celeb64", "AFHQ256", "AFHQ256Exp1", "AFHQ256Exp2", "Low1DMix", "Low2DSymMix", "Low2DASymMix", "Low2DUnimodal", "Low2DBimodal", "Low2DSCurve"]
    if args.log_name in protected_dirs:
        raise Exception(f"Log name {args.log_name} is protected, please choose another one")

    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("logs/training"):
        os.mkdir("logs/training")

    # Remove if older than 30 seconds 
    # If not, keep the folder as it contains data created by SLURM
    if os.path.exists(args.log_dir) and not RUNNING_ON_SLURM:
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
    subdirs = ["models", "samples"]
    [os.makedirs(args.log_dir + "/" + subdir) for subdir in subdirs]

    # Load pretrained args
    if args.pretrained_dirname is not None:
        print("Loading pretrained model")
        # Load pretrained args and merge with current args, prioritizing specified args
        prioritized_args = [e[2:] for e in sys.argv if e.startswith("--")]  + ["log_dir", "log_name", "pretrained_dirname", "pretrained_model"] # Command line args and some important args
        with open(f"logs/training/{args.pretrained_dirname}/args.json", "r") as f:
            pretrained_args = json.load(f)
        for key, value in pretrained_args.items():
            if key not in prioritized_args: 
                setattr(args, key, value)

    if args.dataset == "gaussian_mixture":
        assert args.mus is not None, "Please specify mus for gaussian mixture dataset"
        assert args.sigmas is not None, "Please specify sigmas for gaussian mixture dataset"

    # Set up dataset and device
    dataset = get_dataset(args.dataset, root="src/data", train=True, download=True, mus=args.mus, sigmas=args.sigmas)
    if args.debug_slice is not None: # * DEBUG
        if isinstance(dataset, torch.utils.data.TensorDataset):
            dataset.tensors = tuple(t[:args.debug_slice] for t in dataset.tensors)
        elif isinstance(dataset, torchvision.datasets.VisionDataset):
            dataset.data = dataset.data[:args.debug_slice]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model and trainer
    data_shape = dataset.__getitem__(0)[0].shape


    if args.model_type == "unet":
        assert len(data_shape) == 3, "UNet only supports 3D data (C, H, W)"
        c, h, w = data_shape
        net = UNet(in_channels=c, out_channels=c, data_shape=data_shape, time_dim=256, unet_start_channels=args.unet_start_channels, unet_down_factors=args.unet_down_factors, unet_bot_factors=args.unet_bot_factors, unet_use_attention=args.unet_use_attention)

    elif args.model_type == "single_dim_net":
        assert len(data_shape) == 1, "SingleDimNet only supports 1D data"
        net = SingleDimNet(data_shape[0], data_shape[0], data_shape=data_shape, n_T=args.n_T)
        
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    
    # Create file for writing losses
    with open(f"{args.log_dir}/losses.csv", "w") as f:
        f.write("epoch,loss\n")
    # Create file for reporting lr reductions
    with open(f"{args.log_dir}/lr_reductions.csv", "w") as f:
        f.write("epoch,lr_before,lr_after\n")

    denoiser = DDPM(net, betas=(args.beta1, args.beta2), n_T=args.n_T, schedule_name=args.scheduler)    
    trainer = Trainer(denoiser, dataloader, device, args, do_logging_and_saving=True)

    # Load pretrained model
    if args.pretrained_dirname is not None:
        trainer.load_pretrained(args.pretrained_dirname)

    if args.load_only_models is not None:
        trainer.load_pretrained(args.load_only_models)

    args.data_shape = data_shape # * Additional arg
    # Save args
    with open(args.log_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", default="train_test", help="The directory to log in", type=str)
    parser.add_argument("--dataset", default="mnist", help="The dataset to use", type=str)

    parser.add_argument("--epochs", default=1000, help="The number of epochs to train for", type=int)
    parser.add_argument("--save_interval", default=20, help="The number of epochs between saving models", type=int)

    parser.add_argument("--batch_size", default=64, help="The batch size", type=int)
    parser.add_argument("--lr", default=3e-4, help="The learning rate", type=float)

    parser.add_argument("--ema_decay", default=0.999, help="The decay for the exponential moving average", type=float)

    # * Diffusion specifics
    parser.add_argument("--n_T", default=1000, help="The number of diffusion steps", type=int)
    parser.add_argument("--beta1", default=1e-4, help="Beta1 for diffusion", type=float)
    parser.add_argument("--beta2", default=0.02, help="Beta2 for diffusion", type=float)
    parser.add_argument("--scheduler", default="linear", help="The scheduler to use", type=str, choices=["linear", "cosine", "scaled_linear"])

    # * Model
    parser.add_argument("--model_type", default="unet", help="The model architecture to use", type=str, choices=["unet", "single_dim_net"])

    # * UNet specifics
    parser.add_argument("--unet_start_channels", default=64, help="The number of channels in the first layer of the UNet", type=int)
    parser.add_argument("--unet_down_factors", nargs="+", default=[2, 4, 4], help="The multiplication of channels when downsampling in the UNet", type=int)
    parser.add_argument("--unet_bot_factors", nargs="+", default=[8, 8, 4], help="The multiplication of channels during the bottleneck layers in the UNet", type=int)
    parser.add_argument("--unet_use_attention", default=False, help="Not supported: Whether to use attention in the UNet", type=str_to_bool)

    # * Pretrained specifics
    parser.add_argument("--pretrained_dirname", default=None, help="The name of the directory to load pretrained models from", type=str)
    
    parser.add_argument("--load_only_models", default=None, help="Name of directory for model states to load, and ignore other arguments from the pretrained session")

    # * Debugging specifics
    parser.add_argument("--debug", default=False, help="Whether to run in debug mode. Activates tqdm", type=str_to_bool)
    parser.add_argument("--debug_slice", default=None, help="The slice to use on the dataset for debugging", type=int)

    # * Used in the case of the dataset being a mixture of Gaussians (only supports 1D and 2D specifications)
    parser.add_argument("--mus", nargs="+", default=None, type=lambda x: [float(i) for i in x.split()], help="The means of the Gaussians. Write '--mus 'X'' for a univar single. Write '--mus X Y' for a univar double. Write '--mus 'X1 Y1' 'X2 Y2'' for a bivar double")
    # Bear with me here 
    parser.add_argument("--sigmas", nargs="+", default=None, type=lambda x: [[float(j) for j in i.split(',')] for i in x.split(':')], help="The stds/covariance of the Gaussians. Write '--sigmas X' for univar single. Write '--sigmas X Y' for univar double Write. For bivariate single, write for example '--sigmas 1,0:0,1'. For bivariate double, write for example '--sigmas 1,0:0,1 1,0:0,1'") 

    interface = parser.format_help()
    print(interface)
    args = parser.parse_args()
    print("Running with args: ", args)
    
    run_training(args)
            


