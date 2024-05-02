import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        # self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

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
        self.out_layer = nn.Sequential(
            # nn.Conv2d(32, 32, kernel_size = 3,  padding = 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace = True),
            # nn.Conv2d(32, 32, kernel_size = 3,  padding = 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace = True),
            # nn.Conv2d(32, in_channels, kernel_size = 1)
            nn.Conv2d(32, in_channels, kernel_size = 1)
        )
        # self.out_layer = DoubleConv(32, in_channels, cond_dim)

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
