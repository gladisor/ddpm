import torch
import matplotlib.pyplot as plt

from unet import UNet
from ddpm.data import PokemonImageDataset
from ddpm.sampler import DiffusionSampler

if __name__ == '__main__':

    model = UNet(4, 32).cuda()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    data = PokemonImageDataset('data/pokemon', 64)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 4, shuffle = True, drop_last = True)

    sampler = DiffusionSampler(300).cuda()

    x_0, y = next(iter(dataloader))
    x_0 = x_0.cuda()

    T = torch.ones(4).long().cuda() * 299
    x_T, noise = sampler(x_0, T)

    X = x_T.clone()

    num_images = 10
    step_size = int(300 / num_images)

    fig, ax = plt.subplots(4, 2 + num_images, figsize = (20, 20))
    ## plot original image
    ax[0, 0].imshow(data.data_to_image(x_0[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[1, 0].imshow(data.data_to_image(x_0[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[2, 0].imshow(data.data_to_image(x_0[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[3, 0].imshow(data.data_to_image(x_0[3, :,  :, :].detach().cpu().permute(1, 2, 0)))

    ax[0, 1].imshow(data.data_to_image(x_T[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[1, 1].imshow(data.data_to_image(x_T[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[2, 1].imshow(data.data_to_image(x_T[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
    ax[3, 1].imshow(data.data_to_image(x_T[3, :,  :, :].detach().cpu().permute(1, 2, 0)))

    for i in reversed(range(300)):
        X = sampler.reverse_sample(model, X, torch.ones(4).long().cuda() * i)
        X = X.clamp(-1.0, 1.0)
        print(i)

        if i % step_size == 0:
            ax[0, 2 + num_images-1 - int(i / step_size)].imshow(data.data_to_image(X[0, :,  :, :].detach().cpu().permute(1, 2, 0)))
            ax[1, 2 + num_images-1 - int(i / step_size)].imshow(data.data_to_image(X[1, :,  :, :].detach().cpu().permute(1, 2, 0)))
            ax[2, 2 + num_images-1 - int(i / step_size)].imshow(data.data_to_image(X[2, :,  :, :].detach().cpu().permute(1, 2, 0)))
            ax[3, 2 + num_images-1 - int(i / step_size)].imshow(data.data_to_image(X[3, :,  :, :].detach().cpu().permute(1, 2, 0)))




    fig.savefig('test.png')
