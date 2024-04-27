from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
## https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

def extract(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    '''
    v: vector of values
    t: integer indexes
    '''
    return v.gather(0, t)[:, None, None, None]

# def linear_beta_schedule(timesteps: int, start: float, end: float):
#     return torch.linspace(start, end, timesteps)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionSampler(nn.Module):
    '''
    This is a tool for stepping images to a particular point in time in the forward diffusion process.
    It takes in a noise schecule and number of timesteps.
    '''
    # def __init__(self, timesteps: int, schedule: Callable = linear_beta_schedule, start: float = 0.0001, end: float = 0.02):
    def __init__(self, timesteps: int, schedule: Callable = linear_beta_schedule):
        super().__init__()
        betas = schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0) ## ensure that the first alpha cumprod is 1.0, remove the last

        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alphas_cumprod',  alphas_cumprod, persistent=False)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev, persistent=False)
    
    @torch.no_grad()
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_0: (B x C x H x W) a batch of images
        t: (B,) timestep for each image in batch
        """
        alpha_bar = self.alphas_cumprod.gather(0, t)[:, None, None, None]
        noise = torch.randn_like(x_0)

        x_t = alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * noise
        return x_t, noise
    
    @torch.no_grad()
    def reverse_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        epsilon = model(x_t, t)
        
        alpha = self.alphas.gather(0, t)[:, None, None, None]
        alpha_bar = self.alphas_cumprod.gather(0, t)[:, None, None, None]
        alpha_bar_prev = self.alphas_cumprod_prev.gather(0, t)[:, None, None, None]
        beta = self.betas.gather(0, t)[:, None, None, None]
        first_timestep = (t > 0)[:, None, None, None]

        z = torch.randn_like(x_t)
        # var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        var = beta
        
        mu = (1.0 / alpha.sqrt()) * (x_t - epsilon * beta / (1.0 - alpha_bar).sqrt())
        return (mu + first_timestep * var.sqrt() * z)
    
    # @torch.no_grad()
    # def reverse_sample(self, epsilon, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    #     # epsilon = model(x_t, t)
    #     alpha = self.alphas.gather(0, t)[:, None, None, None]
    #     alpha_bar = self.alphas_cumprod.gather(0, t)[:, None, None, None]
    #     alpha_bar_prev = self.alphas_cumprod_prev.gather(0, t)[:, None, None, None]
    #     beta = self.betas.gather(0, t)[:, None, None, None]
    #     first_timestep = (t > 0)[:, None, None, None]

    #     z = torch.randn_like(x_t)
    #     var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        
    #     mu = (1.0 / alpha.sqrt()) * (x_t - epsilon * beta / (1.0 - alpha_bar).sqrt())
    #     return mu + first_timestep * var.sqrt() * z
    
if __name__ == '__main__':
    sampler =  DiffusionSampler(1000)