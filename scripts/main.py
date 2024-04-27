import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from unet import UNet
from ddpm.data import PokemonImageDataset
from ddpm.sampler import DiffusionSampler
from ddpm.utils import write_image

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## system hyperparameters
    batch_size = 256
    timesteps = 1000
    in_channels = 4
    time_emb_dim = 32
    # noise_start = 0.0001

    data = PokemonImageDataset('data/pokemon')
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)

    sampler = DiffusionSampler(timesteps).to(device)

    model = UNet(in_channels, time_emb_dim).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.train()
    
    optimizer =  torch.optim.Adam(model.parameters(), lr = 0.0005)

    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            x_0, _ = batch
            x_0 = x_0.to(device)

            t = torch.randint(0, timesteps, (batch_size,)).to(device)
            x_t, noise = sampler(x_0, t)

            optimizer.zero_grad()
            loss = F.l1_loss(model(x_t, t), noise)
            loss.backward()
            optimizer.step()

            l = loss.detach().item()
            print(l)

    torch.save(model.state_dict(), 'model.pt')