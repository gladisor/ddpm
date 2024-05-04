import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import tqdm

from ddpm.data import build_image_transform
from ddpm.sampler import DiffusionSampler
from ddpm.model import UNet

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train_data = ImageFolder('data/pokemon', build_image_transform())
    # test_data = ImageFolder('data/pokemon', build_image_transform())
    train_data = ImageFolder('data/birds/train', build_image_transform())
    test_data = ImageFolder('data/birds/test', build_image_transform())
    print(f'Train length = {len(train_data)}, Val length = {len(test_data)}')
    
    ## system hyperparameters
    batch_size = 100
    timesteps = 1000
    in_channels = 3
    time_emb_dim = 64
    epochs = 100
    result_path = 'results/birds/custom/'

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 8, shuffle = True, drop_last = True)
    
    sampler = DiffusionSampler(timesteps).to(device)
    model = UNet().to(device)
    model.train()

    x0, c = next(iter(testloader))
    x0 = x0.to(device)
    # sampler.plot_forward(x0, 10, path = 'forward.png')
    t = torch.randint(0, timesteps, (x0.shape[0],)).long().to(device)
    y = model(x0, t)

    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # scheduler = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, epochs)

    for epoch in range(epochs):
        
        model.train()
        for i, batch in tqdm.tqdm(enumerate(trainloader)):
            x0, _ = batch
            x0 = x0.to(device)

            t = torch.randint(0, timesteps, (batch_size,)).to(device)
            x_t, noise = sampler.step(x0, t)

            opt.zero_grad()
            loss = F.l1_loss(model(x_t, t), noise)
            loss.backward()
            opt.step()

            l = loss.detach().item()

        # scheduler.step()
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