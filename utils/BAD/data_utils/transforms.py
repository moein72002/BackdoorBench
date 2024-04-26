import torchvision
from torchvision import transforms

normal_transform = transforms.Compose(
    [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])

celeba_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

bw_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    ])
