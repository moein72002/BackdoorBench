from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
import os

class CelebA(Dataset):
    def __init__(self, data_root, split, transform = None):
        self.dataset = torchvision.datasets.CelebA(root=data_root, split=split, target_type="attr", download=False)
        self.list_attributes = [18, 31, 21]
        self.transform = transform
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        if self.transform is not None:
            input = self.transform(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

class SingleLabelDataset(Dataset):
    # defining values in the constructor
    def __init__(self, label, dataset, transform = None, lim=None):
        self.data = [dataset[i][0] for i in range(len(dataset) if lim is None else min(lim, len(dataset)))]

        self.label = label

    # Getting the data samples
    def __getitem__(self, idx):
        image = self.data[idx]
        return image, self.label

    # Getting data size/length
    def __len__(self):
        return len(self.data)


class GaussianDataset(Dataset):
    def __init__(self, label, num_samples=3000, size=32):
        self.data = torch.randn((num_samples,3,size,size))
        self.label = label
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label

class BlankDataset(Dataset):
    def __init__(self, label, num_samples=3000, size=32, color=0):
        self.data = torch.ones((num_samples,3,size,size)) * color
        self.label = label
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label