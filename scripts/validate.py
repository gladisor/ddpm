import math

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import tqdm
import lightning as L

from ddpm.data import build_image_transform
from ddpm.sampler import DiffusionSampler
from ddpm.model import UNet

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = ImageFolder('data/birds/train', build_image_transform())
    test_data = ImageFolder('data/birds/test', build_image_transform())
    print(f'Train length = {len(train_data)}, Val length = {len(test_data)}')
    
    ## system hyperparameters
    batch_size = 100
    timesteps = 1000
    in_channels = 3
    time_emb_dim = 64
    epochs = 100
    result_path = 'results/birds/rms_norm_custom_unet/'

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 8, shuffle = True, drop_last = True)

    ## define core components of the model
    sampler = DiffusionSampler(timesteps).to(device)
    model = UNet().to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(trainloader):
            x0, _ = batch
            x0 = x0.to(device)

            t = torch.randint(0, timesteps, (batch_size,)).to(device)
            x_t, noise = sampler.step(x0, t)

            opt.zero_grad()
            loss = F.l1_loss(model(x_t, t), noise)
            loss.backward()
            opt.step()
            l = loss.detach().item()
            print(f'Batch: {i}, Loss: {l}')

            if math.isnan(l):
                print('NAN LOSS DETECTED!')
                torch.save(model.state_dict(), result_path + f'model_epoch={epoch}_NAN_DETECTED.pt')
                break

        model.eval()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), result_path + f'model_epoch={epoch}.pt')

        x0, _ = next(iter(testloader))
        x0 = x0.to(device)
        t = torch.ones(8).long().to(device) * sampler.timesteps-1

        ## reverse noised true images
        xt, _ = sampler.step(x0, t)
        sampler.plot_reverse(model, xt, 5, result_path + f'reverse_epoch={epoch}.png')
        ## novel sampled images
        xt = torch.randn_like(x0)
        sampler.plot_reverse(model, xt, 5, result_path + f'novel_reverse_epoch={epoch}.png')