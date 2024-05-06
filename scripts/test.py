import torch
from torchvision.datasets import ImageFolder

from ddpm.data import build_image_transform
from ddpm.sampler import DiffusionSampler

if __name__ == '__main__':
    test_data = ImageFolder('data/birds/test', build_image_transform())
    
    testloader = torch.utils.data.DataLoader(
        test_data, 
        batch_size = 2, 
        shuffle = True, 
        drop_last = True)
    
    x0, c = next(iter(testloader))
    x0 = x0.to(device)

    print(x0.shape)
