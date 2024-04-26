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
    batch_size = 128
    timesteps = 300

    in_channels = 4
    time_emb_dim = 32

    data = PokemonImageDataset('data/pokemon', 64)
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)

    sampler = DiffusionSampler(timesteps).to(device)

    model = UNet(in_channels, time_emb_dim).to(device)
    # model.load_state_dict(torch.load('model.pt'))
    model.train()
    print(model)
    
    optimizer =  torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(1000):
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

    model.eval()
    x_0, _ = next(iter(dataloader))
    x_0 = x_0.to(device)

    T = torch.ones(batch_size).long().to(device)  * 1

    x_T, noise = sampler(x_0, T)

    x_hat = sampler.reverse_sample(model, x_T, T)
    # x_hat = sampler.reverse_sample(noise, x_T, T)

    print(T)
    print((x_0 - x_hat).max())
    print((x_0 - x_hat).min())

    fig, ax = plt.subplots(4, 3, figsize = (10, 10))
    ## plot original image
    ax[0, 0].imshow(data.data_to_image(x_0[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[1, 0].imshow(data.data_to_image(x_0[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[2, 0].imshow(data.data_to_image(x_0[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[3, 0].imshow(data.data_to_image(x_0[3, :,  :, :].detach().cpu().permute(1, 2, 0)))

    ## plot one step corrupted image
    ax[0, 1].imshow(data.data_to_image(x_T[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[1, 1].imshow(data.data_to_image(x_T[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[2, 1].imshow(data.data_to_image(x_T[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[3, 1].imshow(data.data_to_image(x_T[3, :,  :, :].detach().cpu().permute(1, 2, 0)))

    ## plot reverse sampled (reconstructed) image
    ax[0, 2].imshow(data.data_to_image(x_hat[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[1, 2].imshow(data.data_to_image(x_hat[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[2, 2].imshow(data.data_to_image(x_hat[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[3, 2].imshow(data.data_to_image(x_hat[3, :,  :, :].detach().cpu().permute(1, 2, 0)))
    fig.savefig('image.png')



