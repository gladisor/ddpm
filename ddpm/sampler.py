from typing import Callable
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02):
    return torch.linspace(start, end, timesteps)

class ForwardDiffusionSampler:
    """
    This is a tool for stepping images to a particular point in time in the forward diffusion process.
    It takes in a noise schecule and number of timesteps.
    """
    def __init__(self, schedule: Callable, timesteps: int):
        self.betas = schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.0) ## ensure that the first alpha cumprod is 1.0, remove the last
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_varience = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def sample_single(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        x_0: (C x H x W) image
        t: an integer timestep to step the image to
        """
        alpha_bar = self.alphas_cumprod_prev[t]
        noise = torch.randn_like(x_0)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise
    
if __name__ == '__main__':
    sampler =  ForwardDiffusionSampler(linear_beta_schedule, 1000)