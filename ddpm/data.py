from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import pandas as pd

from ddpm.utils import build_image_to_data_transform, build_data_to_image_transform

class PokemonImageDataset(Dataset):
    def __init__(self, root: str,  img_size: int = 64) -> None:
        super().__init__()

        self.root = Path(root)
        self.data = pd.read_csv(self.root / 'pokemon.csv')
        self.pokemon = self.data['Name'].values

        self.image_to_data = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0)
            ])
        
        self.data_to_image = transforms.Compose([
            transforms.Lambda(lambda x: 255.0 * (x + 1.0) / 2.0),
            transforms.Lambda(lambda x: x.type(torch.ByteTensor))
            ])
        
        images = [
            self.image_to_data(torchvision.io.read_image(str(self.root / 'images' / f'{p}.png'))) for p in self.pokemon
        ]

        self.images = torch.stack(images)

    def __len__(self) -> int:
        return len(self.pokemon)
    
    def __getitem__(self, idx: int):
        return self.images[idx, ...], self.pokemon[idx]

if __name__ == '__main__':
    data = PokemonImageDataset('data/pokemon')
    loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)

    x, y = next(iter(loader))
    print(x)
    print(x.shape)