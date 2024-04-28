from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import tqdm

def negative_one_to_one(x: Tensor) -> Tensor:
    return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0

def viewable(x: Tensor) -> Tensor:
    return ((x + 1.) / 2.).permute(1, 2, 0).clamp(0., 1.)

class ImageDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()

        p = Path(root)
        self.paths = list(p.glob('*.png'))

        image_size = 64
        self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(negative_one_to_one)
                ])
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, i):
        img = Image.open(str(self.paths[i])).convert('RGB')
        x = self.transform(img)
        return x
    
def extract(v: Tensor, t: Tensor) -> Tensor:
    '''
    v: vector of values
    t: integer indexes
    '''
    return v.gather(0, t)[:, None, None, None]

def linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    linear schedule, proposed in original ddpm paper
    '''
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionSampler(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()

        ## computing scheduling parameters
        beta = linear_beta_schedule(timesteps)
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
    
    @torch.inference_mode()
    def step(self, x0: Tensor, t: Tensor):
        noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t)
        xt = alpha_bar.sqrt() * x0 + (1. - alpha_bar).sqrt() * noise
        return xt, noise
    
    @torch.inference_mode()
    def reverse_step(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:

        beta            = extract(self.beta, t)
        alpha           = extract(self.alpha, t)
        alpha_bar       = extract(self.alpha_bar, t)
        alpha_bar_prev  = extract(self.alpha_bar_prev, t)
        not_first_timestep = (t > 1)[:, None, None, None]

        epsilon = model(xt, t)
        z = torch.randn_like(xt)
        var = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        mu = (1.0 / alpha.sqrt()) * (xt - epsilon * beta / (1.0 - alpha_bar).sqrt())
        return (mu + not_first_timestep * var.sqrt() * z)
    
    @torch.inference_mode()
    def denoise(self, model: nn.Module, xT: Tensor, return_sequence = False) -> Tensor:

        xt = xT.clone()

        if return_sequence:
            sequence = [xt]

        for i in tqdm.tqdm(reversed(range(self.timesteps))):
            t = torch.ones(xT.shape[0]).long().to(self.device) * i
            xt = self.reverse_step(model, xt, t)
            
            if return_sequence:
                sequence.append(xt)

        if return_sequence:
            return sequence
        else:
            return xt

    @torch.inference_mode()
    def plot_forward(self, x0: Tensor, num_images: int):
        batch_size = x0.shape[0]
        timesteps = len(self.beta)

        scale = 2
        fig, ax = plt.subplots(batch_size, num_images + 1, figsize = (scale*num_images, scale*batch_size))
        ax[0, 0].set_title('Original')
        for j in range(batch_size):
            ax[j, 0].imshow(viewable(x0[j, ...]))

        for i in range(1, num_images):
            T = i * int(timesteps / num_images)
            t = torch.ones(batch_size).long().to(self.device) * T
            xt, _ = self.step(x0, t)
            ax[0, i].set_title(str(T))
            for j in range(batch_size):
                ax[j, i].imshow(viewable(xt[j, ...]))

        for j in range(batch_size):
            T = timesteps - 1
            t = torch.ones(batch_size).long().to(self.device) * T
            xt, _ = self.step(x0, t)

            ax[0, -1].set_title(str(T))
            for j in range(batch_size):
                ax[j, -1].imshow(viewable(xt[j, ...]))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].axis('off')
        
        fig.tight_layout()
        fig.savefig('forward.png')
        plt.close(fig)
        return None
    
    # @torch.inference_mode()
    # def plot_reverse(self, x0: Tensor, num_images: int):
    #     batch_size = x0.shape[0]
    #     timesteps = len(self.beta)

    #     scale = 2
    #     fig, ax = plt.subplots(batch_size, num_images + 1, figsize = (scale*num_images, scale*batch_size))
    #     ax[0, 0].set_title('Original')
    #     for j in range(batch_size):
    #         ax[j, 0].imshow(viewable(x0[j, ...]))

    #     for i in range(1, num_images):
    #         T = i * int(timesteps / num_images)
    #         t = torch.ones(batch_size).long().to(self.device) * T
    #         xt, _ = self.step(x0, t)
    #         ax[0, i].set_title(str(T))
    #         for j in range(batch_size):
    #             ax[j, i].imshow(viewable(xt[j, ...]))

    #     for j in range(batch_size):
    #         T = timesteps - 1
    #         t = torch.ones(batch_size).long().to(self.device) * T
    #         xt, _ = self.step(x0, t)

    #         ax[0, -1].set_title(str(T))
    #         for j in range(batch_size):
    #             ax[j, -1].imshow(viewable(xt[j, ...]))

    #     for i in range(ax.shape[0]):
    #         for j in range(ax.shape[1]):
    #             ax[i, j].axis('off')
        
    #     fig.tight_layout()
    #     fig.savefig('forward.png')
    #     plt.close(fig)
    #     return None

class SinusoidalEmbeddings(nn.Module):
    '''
    A module for converting a batch of integer timesteps into a batch of positional embeddings.
    '''
    def __init__(self, dim: int):
        super().__init__()

        assert dim % 2 == 0, 'Argument dim must be even.'
        dim = torch.arange(dim)
        self.register_buffer('dim', dim, persistent = False)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb1 = 10000 ** (self.dim[0::2] / len(self.dim))
        emb2 = 10000 ** (self.dim[1::2] / len(self.dim))
        emb = torch.stack([
            (time[:, None] / emb1[None, :]).sin(),
            (time[:, None] / emb2[None, :]).cos()],
        ).permute(1, 2, 0).reshape(time.shape[0], emb1.shape[0] + emb2.shape[0])
        return emb
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv(self.upsample(x)), z], dim = 1)
        return y

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, in_channels),
            nn.Unflatten(1, (in_channels, 1, 1))
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x, c):
        return self.conv(self.cond_mlp(c) + x)
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, cond_dim: int):
        super().__init__()

        self.emb = nn.Sequential(
            SinusoidalEmbeddings(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True)
        )

        self.in_layer = DoubleConv(in_channels, 32, cond_dim)
        self.down1 = DownBlock(32, 32)
        self.down1_conv = DoubleConv(32, 64, cond_dim)
        self.down2 = DownBlock(64, 64)
        self.down2_conv = DoubleConv(64, 128, cond_dim)
        self.down3 = DownBlock(128, 128)
        self.down3_conv = DoubleConv(128, 256, cond_dim)

        self.up1 = UpBlock(256, 128)
        self.up1_conv = DoubleConv(256, 128, cond_dim)
        self.up2 = UpBlock(128, 64)
        self.up2_conv = DoubleConv(128, 64, cond_dim)
        self.up3 = UpBlock(64, 32)
        self.up3_conv = DoubleConv(64, 32, cond_dim)
        self.out_layer = nn.Conv2d(32, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.emb(t)

        x1 = self.in_layer(x, t_emb)
        x2 = self.down1_conv(self.down1(x1), t_emb)
        x3 = self.down2_conv(self.down2(x2), t_emb)
        x4 = self.down3_conv(self.down3(x3), t_emb)

        y = self.up1_conv(self.up1(x4, x3), t_emb)
        y = self.up2_conv(self.up2(y, x2), t_emb)
        y = self.up3_conv(self.up3(y, x1), t_emb)

        return self.out_layer(y)


if __name__ == '__main__':

    data = ImageDataset('data/pokemon/images')

    batch_size = 6
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)

    x0 = next(iter(dataloader))
    plt.imshow(viewable(x0[0, ...]))
    plt.savefig('img.png')
    plt.close()

    sampler = DiffusionSampler(100)
    sampler.plot_forward(x0, 10)

    model = UNet(3, 64)
    model.train()
    model.eval()

    t = torch.ones(batch_size).long() * sampler.timesteps-1
    xt, noise = sampler.step(x0, t)

    print(t)
    x0_hat = sampler.denoise(model, xt)
    print(x0_hat)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(viewable(x0[0, ...]))
    ax[1].imshow(viewable(xt[0, ...]))
    ax[2].imshow(viewable(x0_hat[0, ...]))
    fig.savefig('compare.png')
    plt.close(fig)