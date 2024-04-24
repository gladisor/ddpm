import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def build_image_to_data_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0)
        ])

def build_data_to_image_transform():
    return transforms.Compose([
        transforms.Lambda(lambda x: 255.0 * (x + 1.0) / 2.0),
        transforms.Lambda(lambda x: x.type(torch.ByteTensor))
        ])

def show_image(img: torch.Tensor):
    """
    Shows a tensor image to the screen.

    img: (C x H x W)
    """
    plt.imshow(torch.permute(img, (1, 2, 0)).numpy())
    plt.show()

def write_image(img: torch.Tensor, path: str):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.permute((1, 2, 0)))
    fig.savefig(path)
    plt.close(fig)