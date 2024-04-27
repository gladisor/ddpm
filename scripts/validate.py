from torchvision import transform
from PIL import Image

if __name__ == '__main__':

    image_size = 64

    transform = transform.Compose([
            transform.Resize(image_size),
            transform.RandomHorizontalFlip(),
            transform.CenterCrop(image_size),
            transform.ToTensor()
        ])
    
    img = Image.open('data/pokemon/images/abomasnow.png')

    print(img)

