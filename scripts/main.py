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
    batch_size = 4
    timesteps = 300

    in_channels = 4
    time_emb_dim = 32

    data = PokemonImageDataset('data/pokemon', 64)
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)

    sampler = DiffusionSampler(timesteps, start = 0.00001).to(device)

    model = UNet(in_channels, time_emb_dim).to(device)
    # model.load_state_dict(torch.load('model.pt'))
    model.train()
    
    optimizer =  torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(10):
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

    T = torch.ones(batch_size).long().to(device) * 0

    x_T, noise = sampler(x_0, T)

    x_hat = sampler.reverse_sample(model, x_T, T)

    # fig, ax = plt.subplots(batch_size, 4, figsize = (10, 10))

    # for i in range(batch_size):
    #     ax[i, 0].imshow(data.data_to_image(x_0[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    # for i in range(batch_size):
    #     ax[i, 1].imshow(data.data_to_image(x_T[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    # for i in range(batch_size):
    #     ax[i, 2].imshow(data.data_to_image(noise[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    # for i in range(batch_size):
    #     ax[i, 3].imshow(data.data_to_image(x_hat[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    # fig.savefig('noise.png')











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



