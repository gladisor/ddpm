import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from unet import UNet
from ddpm.data import PokemonImageDataset
from ddpm.sampler import DiffusionSampler

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## system hyperparameters
    batch_size = 128
    timesteps = 200
    in_channels = 4
    time_emb_dim = 64

    data = PokemonImageDataset('data/pokemon')
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)

    sampler = DiffusionSampler(timesteps).to(device)

    model = UNet(in_channels, time_emb_dim).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.train()
    
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)

    epochs = 300
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, epochs)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            x_0, _ = batch
            x_0 = x_0.to(device)

            t = torch.randint(0, timesteps, (batch_size,)).to(device)
            x_t, noise = sampler(x_0, t)

            opt.zero_grad()
            loss = F.l1_loss(model(x_t, t), noise)
            loss.backward()
            opt.step()

            l = loss.detach().item()
            print(l)
        
        scheduler.step()

    torch.save(model.state_dict(), 'model.pt')