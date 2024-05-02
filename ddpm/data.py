from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from ddpm.utils import negative_one_to_one

def build_image_transform(size: int = 128):
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(negative_one_to_one)
        ])