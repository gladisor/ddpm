import torch
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from ddpm.data import build_image_transform
from ddpm.sampler import DiffusionSampler
from ddpm.utils import viewable

## https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1841s
## https://huggingface.co/blog/annotated-diffusion
from torch import nn, Tensor, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import math

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.rearrange(x))
    
def exists(x):
    return x is not None

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale: Tensor = None, shift: Tensor = None) -> Tensor:
        x = self.norm(self.conv(x))
        if exists(scale) and exists(shift):
            x = x * (scale + 1.0) + shift
        return self.act(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
        
        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x: Tensor, time_emb: Tensor):
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale, shift)
        return self.block2(h) + self.skip(x)

class PreNorm(nn.Module):
    def __init__(self, layer: nn.Module, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(self.norm(x))
    
## https://huggingface.co/blog/annotated-diffusion
class Attention(nn.Module):
    '''
    Computes full pixelwise attention.
    '''
    def __init__(self, in_channels: int, heads: int = 2, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        hidden_dim = heads * dim_head
        self.qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size = 1, bias = False)
        self.output = nn.Conv2d(hidden_dim, in_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        ## compute the queries, keys, and values of the incoming feature maps
        q, k, v = torch.chunk(self.qkv(x), 3, dim = 1)
        ## takes the channel dimention and reshapes into heads, channels
        ## also flattens the feature maps into vectors
        ## the shape is now b h c d where d is dim_head
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h = self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h = self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h = self.heads)
        q = q * self.scale

        ## multiplication of the query and key matrixes for each head
        ## this is a pixelwise comparison since we flattened the feature maps
        sim = einsum('b h c i, b h c j -> b h i j', q, k)
        ## subtract the maximum value of each row
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        ## make each row into a probability distribution
        attn = sim.softmax(dim = -1)
        ## weight the values according to the rows of the attention matrix
        y = einsum('b h i j, b h d j -> b h i d', attn, v)
        ## reshape the weighted values back into feature maps
        y = rearrange(y, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        ## return the feature maps but compress the channel dimention back to the original size
        return self.output(y)
    
## https://huggingface.co/blog/annotated-diffusion
class LinearAttention(nn.Module):
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head
        self.qkv = nn.Conv2d(in_channels, 3*hidden_dim, kernel_size = 1, bias = False)
        self.output = nn.Conv2d(hidden_dim, in_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = 1)

        ## takes the channel dimention and reshapes into heads, channels
        ## also flattens the feature maps into vectors
        ## the shape is now b h c d where d is dim_head
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h = self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h = self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h = self.heads)
        ## softmax along dim_head dim
        q = q.softmax(dim = 2) * self.scale
        ## softmax along flattened image dim
        k = k.softmax(dim = 3)
        ## compute comparison betweeen keys and values to produce context.
        ## essentially this is a comparison between each flattened image in the dim_head
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = h, y = w)
        return self.output(out)
    
class UNet(nn.Module()):
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(160)
        )

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = ImageFolder('data/pokemon', build_image_transform())
    test_data = ImageFolder('data/pokemon', build_image_transform())
    # train_data = ImageFolder('data/birds/train', build_image_transform())
    # test_data = ImageFolder('data/birds/test', build_image_transform())
    print(f'Train length = {len(train_data)}, Val length = {len(test_data)}')
    
    ## system hyperparameters
    batch_size = 100
    timesteps = 1000
    in_channels = 3
    time_emb_dim = 64
    epochs = 100
    result_path = 'results/birds/youtube/'

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 8, shuffle = True, drop_last = True)
    sampler = DiffusionSampler(timesteps).to(device)

    x0, c = next(iter(testloader))
    x0 = x0.to(device)

    # sampler.plot_forward(x0, 10, path = 'forward.png')
    t = torch.randint(0, timesteps, (x0.shape[0],)).long().to(device)
    # time_emb_dim = 160
    # emb = SinusoidalEmbeddings(time_emb_dim).to(device)
    # layer = ResBlock(3, 16, time_emb_dim).to(device)
    # down1 = DownBlock(16, 32).to(device)
    # down2 = DownBlock(32, 64).to(device)
    # down3 = DownBlock(64, 128).to(device)

    # attention = Attention(32, 2).to(device)
    # linear_attention = LinearAttention(32, 2).to(device)

    # time_emb = emb(t)
    # y = down1(layer(x0, time_emb))

    model = UNet()

    model.time_mlp(t)

    print(x0.shape)
    print(t.shape)









    # model = Unet(dim = 32).to(device)
    # model.train()

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)


    # # opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # # # scheduler = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, epochs)


    # # for epoch in range(epochs):
        
    # #     model.train()
    # #     for i, batch in tqdm.tqdm(enumerate(trainloader)):
    # #         x0, _ = batch
    # #         x0 = x0.to(device)

    # #         t = torch.randint(0, timesteps, (batch_size,)).to(device)
    # #         x_t, noise = sampler.step(x0, t)

    # #         opt.zero_grad()
    # #         loss = F.l1_loss(model(x_t, t), noise)
    # #         loss.backward()
    # #         opt.step()

    # #         l = loss.detach().item()

    # #     # scheduler.step()
    # #     model.eval()
    # #     torch.save(model.state_dict(), result_path + f'model_epoch={epoch}.pt')

    # #     x0, _ = next(iter(testloader))
    # #     x0 = x0.to(device)

    # #     t = torch.ones(8).long().to(device) * sampler.timesteps-1
    # #     xt, noise = sampler.step(x0, t)
    # #     sampler.plot_reverse(model, xt, 5, result_path + f'reverse_epoch={epoch}.png')

    # #     xt = torch.randn_like(x0)
    # #     sampler.plot_reverse(model, xt, 5, result_path + f'novel_reverse_epoch={epoch}.png')