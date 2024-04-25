import torch
from torch import nn
import numpy as np

class SinusoidalEmbeddings(nn.Module):
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv(self.upsample(x)), z], dim = 1)
        return y
    
class UNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_layer = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(DownBlock(16, 16), DoubleConv(16, 32))
        self.down2 = nn.Sequential(DownBlock(32, 32), DoubleConv(32, 64))

        self.up1 = UpBlock(64, 32)
        self.conv1 = DoubleConv(64, 32)
        self.up2 = UpBlock(32, 16)
        self.conv2 = DoubleConv(32, 16)
        self.out_layer = DoubleConv(16, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        y = self.conv1(self.up1(x3, x2))
        y = self.conv2(self.up2(y, x1))
        return self.out_layer(y)
