import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from ddpm.utils import build_image_to_data_transform, build_data_to_image_transform, write_image
from unet import SinusoidalEmbeddings, UNet

image_to_data = build_image_to_data_transform(120)
data_to_image = build_data_to_image_transform()

img = torchvision.io.read_image("data/pokemon/images/weezing.png")
x_0 = image_to_data(img)[None, :, :, :]

# emb = SinusoidalEmbeddings(256)
# t = torch.arange(1000)
# y = emb(t)
# fig, ax = plt.subplots(1, 1)
# ax.imshow(y.detach(), cmap='viridis')
# fig.savefig("emb.png")

model = UNet(4)
y = model(x_0)
print(x_0.min(), x_0.max())
print(y.min(), y.max())
write_image(data_to_image(y.detach()[0, :, :, :]), "model.png")