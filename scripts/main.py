import torch
from torch import nn
import torchvision

from ddpm.utils import build_image_to_data_transform, build_data_to_image_transform, write_image

image_to_data = build_image_to_data_transform(120)
data_to_image = build_data_to_image_transform()

img = torchvision.io.read_image("data/pokemon/images/weezing.png")
x_0 = image_to_data(img)

model = nn.Sequential(
    nn.Conv2d(4, 64, 2),
    nn.ReLU(),
    nn.Conv2d(64, 4, 2))

y = data_to_image(model(x_0).detach())
write_image(y, 'model.png')