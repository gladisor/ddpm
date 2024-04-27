import torch
import matplotlib.pyplot as plt

from unet import UNet
from ddpm.data import PokemonImageDataset
from ddpm.sampler import DiffusionSampler

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## system hyperparameters
    batch_size = 4
    timesteps = 1000
    in_channels = 4
    time_emb_dim = 32
    # noise_start = 0.0001

    model = UNet(in_channels, time_emb_dim).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    data = PokemonImageDataset('data/pokemon')
    dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)
    sampler = DiffusionSampler(timesteps).to(device)

    ## grab a random batch
    x_0, y = next(iter(dataloader))
    x_0 = x_0.to(device)

    ## sample an image at the final timestep
    T = torch.ones(batch_size).long().to(device) * (timesteps - 1)
    x_T, noise = sampler(x_0, T)

    ## create a deepcopy
    # X = x_T.clone()
    X = torch.randn_like(x_T)

    num_images = 10
    step_size = int(timesteps / num_images)

    fig, ax = plt.subplots(batch_size, 2 + num_images, figsize = (20, 20))
    ## plot original image
    ax[0, 0].set_title('Original')
    for i in range(batch_size):
        ax[i, 0].imshow(data.data_to_image(x_0[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    ## plot noised image
    ax[0, 1].set_title(str(timesteps))
    for i in range(batch_size):
        ax[i, 1].imshow(data.data_to_image(x_T[i, :,  :, :].detach().cpu().permute(1, 2, 0)))

    ## plot the denoising process
    for i in reversed(range(timesteps)):
        X = sampler.reverse_sample(model, X, torch.ones(batch_size).long().cuda() * i)
        X = X.clamp(-1.0, 1.0)
        if i % step_size == 0 or i == 0:
            print(i)
            ax_x_idx = 2 + num_images-1 - int(i / step_size)
            ax[0, ax_x_idx].set_title(str(i))

            for j in range(batch_size):
                ax[j, ax_x_idx].imshow(data.data_to_image(X[j, :,  :, :].detach().cpu().permute(1, 2, 0)))

    fig.savefig('test.png')
