from argparse import ArgumentParser

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import lightning as L

from ddpm.data import build_image_transform
from ddpm.sampler import DiffusionSampler
from ddpm.model import UNet

class LitDDPM(L.LightningModule):
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__()
        self.sampler = DiffusionSampler(timesteps, beta_schedule)
        self.unet = UNet()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters())
    
    def training_step(self, batch, batch_idx: int):
        x0, _ = batch
        b, c, h, w = x0.shape

        t = torch.randint(0, self.sampler.timesteps, (b,), device = self.device)
        x_t, noise = self.sampler.step(x0, t)
        loss = F.l1_loss(self.unet(x_t, t), noise)
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        x0, _ = batch
        b, c, h, w = x0.shape
        val_path = 'results/birds/Lightning/test_linear/'
        ## novel sampled images
        xt = torch.randn_like(x0)
        self.sampler.plot_reverse(self.unet, xt, 5, val_path + f'novel_reverse_step={self.global_step}_device={self.device}.png')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--timesteps', type = int, default = 1000)
    parser.add_argument('--beta_schedule', type = str, default = 'cosine')
    parser.add_argument('--devices', type = int, default = 4)
    parser.add_argument('--epochs', type = int, default = 1000)
    args = parser.parse_args()

    train_data = ImageFolder('data/birds/train', build_image_transform())
    test_data = ImageFolder('data/birds/test', build_image_transform())
    print(f'Train length = {len(train_data)}, Val length = {len(test_data)}')
    
    trainloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = args.batch_size, 
        shuffle = True, 
        drop_last = True, 
        num_workers = 15)
    
    testloader = torch.utils.data.DataLoader(
        test_data, 
        batch_size = 2, 
        shuffle = True, 
        drop_last = True)

    model = LitDDPM(args.timesteps, args.beta_schedule)

    trainer = L.Trainer(
        devices = args.devices,
        accelerator='gpu',
        limit_val_batches = 1,
        max_epochs = args.epochs)
    
    trainer.fit(
        model, 
        train_dataloaders = trainloader,
        val_dataloaders = testloader,
        )
