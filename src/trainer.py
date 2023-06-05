import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from denoiser import Denoiser
from custom_types import TrainingArgsType

import numpy as np

import torch.optim as optim

from ema import EMA
from denoiser import DDPM, DDIM
import tqdm


class Trainer:
    """
    Trainer class for training a denoiser
    """
    def __init__(self, denoiser: Denoiser, dataloader: DataLoader, device, args: TrainingArgsType, do_logging_and_saving: bool = False):

        """
        Args: 
            denoiser: Denoiser object to train
            dataloader: Dataloader for training data
            device: Device to train on
            args: TrainingArgsType object
            do_logging_and_saving: Whether to log and save data
        """

        self.denoiser = denoiser
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.do_logging_and_saving = do_logging_and_saving

        self.ema = EMA(beta=args.ema_decay, model=self.denoiser.model)

        self.optimizer = optim.Adam(self.denoiser.model.parameters(), lr=args.lr)

        self.denoiser.to(device)
        self.denoiser.model.to(device)
        self.ema.ema_model.to(device)

        fig, ax = plt.subplots()
        ax.set_title("alphabar")
        ax.plot(self.denoiser.alphabar.cpu().numpy())
        ax.text(0.95, 0.95, f"alphabar: {self.denoiser.alphabar[-1].item()}", horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        fig.savefig(f"{self.args.log_dir}/alphabar.png") if self.do_logging_and_saving else None
        plt.close(fig)

        self.start_epoch = 0
        # Fetch list of losses from previous training
        if args.pretrained_dirname is not None and self.do_logging_and_saving:
            # Append existing losses and set start epoch
            losses = np.genfromtxt(f"logs/training/{self.args.pretrained_dirname}/losses.csv", delimiter=",")[1:]
            self.start_epoch = int(losses[-1, 0]) + 1
            losses = losses.tolist()
            losses = [(int(epoch), float(loss)) for epoch, loss in losses]
            self.write_losses(losses)


        # For one and two dimensional data, plot the data distribution and save
        if len(self.denoiser.data_shape) == 1 and isinstance(self.dataloader.dataset, torch.utils.data.TensorDataset):
            if self.denoiser.data_shape[0] == 1:
                fig, ax = plt.subplots()
                ax.set_title("data")
                ax.hist(self.dataloader.dataset.tensors[0][:4000].cpu().numpy(), bins=100, density=True)
                fig.savefig(f"{self.args.log_dir}/data.png") if self.do_logging_and_saving else None
                plt.close(fig)
            elif self.denoiser.data_shape[0] == 2:
                fig, ax = plt.subplots()
                ax.set_title("data")
                ax.scatter(self.dataloader.dataset.tensors[0][:4000, 0].cpu().numpy(), self.dataloader.dataset.tensors[0][:4000, 1].cpu().numpy(), s=1)
                ax.scatter(torch.randn(1000).cpu().numpy(), torch.randn(1000).cpu().numpy(), s=1, alpha=0.1)
                fig.savefig(f"{self.args.log_dir}/data.png") if self.do_logging_and_saving else None
                plt.close(fig)
            


    def run_epoch(self):
        """
        Run a single epoch of training
        """
        self.denoiser.train()
        losses = []
        use_tqdm = self.args.debug
        loader = tqdm.tqdm(self.dataloader) if use_tqdm else self.dataloader
        for i, batch in enumerate(loader):
            x = batch[0] if isinstance(batch, list) or isinstance(batch, tuple) else batch
            x = x.to(self.device)
            loss = self.denoiser.forward_process_loss(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.ema is not None:
                self.ema.step_ema(self.denoiser.model)
            
            losses.append(loss.item())
        # print(self.ema.step)
        return np.mean(losses)

    def run(self):
        """
        Run training
        """
        self.denoiser.save_model(f"{self.args.log_dir}/models/saved_model.pt") if self.do_logging_and_saving else None
        torch.save(self.ema.ema_model.state_dict(), f"{self.args.log_dir}/models/ema_model.pt") if self.do_logging_and_saving else None

        losses_not_saved = []
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            loss_scalar = self.run_epoch()
            losses_not_saved.append((epoch, loss_scalar))

            lr_before = self.optimizer.param_groups[0]["lr"]
            # self.scheduler.step(loss_scalar)
            lr_after = self.optimizer.param_groups[0]["lr"]
            self.report_lr_reductions(lr_before, lr_after, epoch)

            if epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1 or epoch in range(11):
                
                if self.do_logging_and_saving:
                    ddim_denoiser = DDIM(self.denoiser.model, betas=(self.args.beta1, self.args.beta2), n_T=self.args.n_T, schedule_name=self.args.scheduler)
                    self.denoiser.save_model(f"{self.args.log_dir}/models/saved_model.pt")
                    self.sample_and_save(ddim_denoiser, epoch=epoch, n=8, device=self.device)

                    # Sample using EMA
                    ema_denoiser = DDIM(self.ema.ema_model, betas=(self.args.beta1, self.args.beta2), n_T=self.args.n_T, schedule_name=self.args.scheduler)
                    ema_denoiser.to(self.device)
                    self.sample_and_save(ema_denoiser, epoch=epoch, n=8, device=self.device, sample_name="ema")

                    torch.save(self.ema.ema_model.state_dict(), f"{self.args.log_dir}/models/ema_model.pt")

                    # Save losses
                    self.write_losses(losses_not_saved)
                losses_not_saved = []

            print("Epoch", epoch, "loss:", loss_scalar)


    def sample_and_save(self, denoiser, epoch, n, device, sample_name=None):
        """
        Sample from the denoiser and save the samples

        Args:
            denoiser: The denoiser to sample from
            epoch: The current epoch
            n: The number of samples to generate
            device: The device to use
            sample_name: The name of the sample
        """
        denoiser.eval()
        denoiser.set_inference_timesteps(30)
        samples = denoiser.sample(n=n, device=device)
        file_name = f"{sample_name}_{epoch}" if sample_name != None else f"{epoch}"
        
        if self.args.model_type == "unet":
            samples = denoiser.sample(n=n, device=device)
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            path = f"{self.args.log_dir}/samples/{file_name}.png"
            vutils.save_image(samples, path, normalize=False, nrow=n//2)

        if self.args.model_type == "single_dim_net":
            if self.args.data_shape[-1] == 1:
                samples = denoiser.sample(n=n*16, device=device)
                samples = samples.cpu().detach().numpy()
                # np.save(f"{self.args.log_dir}/samples/{epoch}.npy", samples)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(samples[:, 0], [0] * len(samples), s=1)
                fig.savefig(f"{self.args.log_dir}/samples/{file_name}.png")
                plt.close(fig)
            elif self.args.data_shape[-1] == 2:
                samples = denoiser.sample(n=n*160, device=device)
                samples = samples.cpu().detach().numpy()
                # np.save(f"{self.args.log_dir}/samples/{epoch}.npy", samples)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(samples[:, 0], samples[:, 1], s=1)
                fig.savefig(f"{self.args.log_dir}/samples/{file_name}.png")
                plt.close(fig)
            elif self.args.data_shape[-1] == 3:
                samples = denoiser.sample(n=n*160, device=device)
                samples = samples.cpu().detach().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1)
                fig.savefig(f"{self.args.log_dir}/samples/{file_name}.png")
                plt.close(fig)


    def write_losses(self, losses):
        """
        Write losses to file
        """
        with open(f"{self.args.log_dir}/losses.csv", "a") as f:
            for epoch, loss in losses:
                f.write(f"{epoch},{loss}\n")
    
    def report_lr_reductions(self, lr_before, lr_after, epoch):
        # Not used
        if lr_before != lr_after:
            with open(f"{self.args.log_dir}/lr_reductions.csv", "a") as f:
                f.write(f"{epoch},{lr_before},{lr_after}\n")

    def load_pretrained(self, pretrained_dirname):
        """
        Load a pretrained model
        """
        self.denoiser.load_model(f"logs/training/{pretrained_dirname}/models/saved_model.pt", device=self.device)
        self.ema.ema_model.load_state_dict(torch.load(f"logs/training/{pretrained_dirname}/models/ema_model.pt", map_location=self.device))
        self.ema.step = self.ema.step_start # EMA should be active from the start

        self.denoiser.to(self.device)
        self.denoiser.model.to(self.device)
        self.ema.ema_model.to(self.device)