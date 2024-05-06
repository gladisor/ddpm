import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt

from ddpm.utils import extract, viewable

def linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    linear schedule, proposed in original ddpm paper
    '''
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionSampler(nn.Module):
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__()

        ## computing scheduling parameters
        if beta_schedule == 'linear':
            beta = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unrecognized beta_schedule.')
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim = 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value = 1.)

        ## adding as non trainable parameters
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

    @property
    def device(self):
        return self.beta.device
    
    @property
    def timesteps(self):
        return len(self.beta)
    
    @torch.no_grad()
    def step(self, x0: Tensor, t: Tensor):
        noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t)
        xt = alpha_bar.sqrt() * x0 + (1. - alpha_bar).sqrt() * noise
        return xt, noise
    
    @torch.no_grad()
    def reverse_step(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:

        beta               = extract(self.beta, t)
        alpha              = extract(self.alpha, t)
        alpha_bar          = extract(self.alpha_bar, t)
        alpha_bar_prev     = extract(self.alpha_bar_prev, t)
        not_first_timestep = (t > 1)[:, None, None, None]

        epsilon = model(xt, t)
        z = torch.randn_like(xt)
        var = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        mu = (1.0 / alpha.sqrt()) * (xt - epsilon * beta / (1.0 - alpha_bar).sqrt())
        return (mu + not_first_timestep * var.sqrt() * z)
    
    @torch.no_grad()
    def denoise(self, model: nn.Module, xT: Tensor, num_images: int = 1) -> Tensor:

        xt = xT.clone()

        if num_images > 1:
            dt = int(self.timesteps / num_images)
            sequence = []

        for i in tqdm.tqdm(reversed(range(self.timesteps))):
            # t = torch.ones(xT.shape[0]).long().to(self.device) * i
            t = torch.ones(xT.shape[0], device = xT.device).long() * i
            xt = self.reverse_step(model, xt, t)

            if num_images > 1 and (i % dt == 0 or i == self.timesteps - 1):
                sequence.append(xt)

        if num_images > 1:
            return sequence
        else:
            return xt

    @torch.no_grad()
    def plot_forward(self, x0: Tensor, num_images: int, path: str):
        batch_size = x0.shape[0]
        timesteps = len(self.beta)

        scale = 2
        fig, ax = plt.subplots(batch_size, num_images + 1, figsize = (scale*num_images, scale*batch_size))
        ax[0, 0].set_title('Original')
        for j in range(batch_size):
            ax[j, 0].imshow(viewable(x0[j, ...]))

        for i in range(1, num_images):
            T = i * int(timesteps / num_images)
            t = torch.ones(batch_size, device = x0.device).long() * T
            # t = torch.ones(batch_size).long() * T
            xt, _ = self.step(x0, t)
            ax[0, i].set_title(str(T))
            for j in range(batch_size):
                ax[j, i].imshow(viewable(xt[j, ...]))

        for j in range(batch_size):
            T = timesteps - 1
            t = torch.ones(batch_size, device = x0.device).long() * T
            # t = torch.ones(batch_size).long() * T
            xt, _ = self.step(x0, t)

            ax[0, -1].set_title(str(T))
            for j in range(batch_size):
                ax[j, -1].imshow(viewable(xt[j, ...]))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].axis('off')
        
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return None
    
    @torch.no_grad()
    def plot_reverse(self, model: nn.Module, xT: Tensor, num_images: int, path: str):
        batch_size = xT.shape[0]

        sequence = self.denoise(model, xT, num_images)

        scale = 2
        fig, ax = plt.subplots(batch_size, num_images + 1, figsize = (scale*num_images, scale*batch_size))

        for i in range(len(sequence)):
            for j in range(batch_size):
                ax[j, i].imshow(viewable(sequence[i][j, ...]))
                ax[j, i].axis('off')
        
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return