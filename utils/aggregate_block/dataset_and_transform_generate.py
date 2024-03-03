'''
This code is based on https://github.com/bboylyg/NAD

The original license:
License CC BY-NC

The update include:
    1. decompose the function structure and add more normalization options
    2. add more dataset options, and compose them into dataset_and_transform_generate

# idea : use args to choose which dataset and corresponding transform you want
'''
import logging
import os
import random
import pickle
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, Image, ImageOps
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder



def get_num_classes(dataset_name: str) -> int:
    # idea : given name, return the number of class in the dataset
    if dataset_name in ["mnist", "cifar10"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tiny':
        num_classes = 200
    elif dataset_name == 'imagenet':
        num_classes = 1000
    elif dataset_name == 'imagenet30':
        num_classes = 30
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    # idea : given name, return the image size of images in the dataset
    if dataset_name == "cifar10":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "gtsrb":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "mnist":
        input_height = 28
        input_width = 28
        input_channel = 1
    elif dataset_name == "celeba":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == 'cifar100':
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == 'tiny':
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == 'imagenet':
        input_height = 224
        input_width = 224
        input_channel = 3
    elif dataset_name == 'imagenet30':
        input_height = 224
        input_width = 224
        input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return input_height, input_width, input_channel


def get_dataset_normalization(dataset_name):
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization


def get_dataset_denormalization(normalization: transforms.Normalize):
    mean, std = normalization.mean, normalization.std

    if mean.__len__() == 1:
        mean = - mean
    else:  # len > 1
        mean = [-i for i in mean]

    if std.__len__() == 1:
        std = 1 / std
    else:  # len > 1
        std = [1 / i for i in std]

    # copy from answer in
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    # user: https://discuss.pytorch.org/u/svd3

    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=std),
        transforms.Normalize(mean=mean,
                             std=[1., 1., 1.]),
    ])

    return invTrans


def get_transform(dataset_name, input_height, input_width, train=True, random_crop_padding=4):
    # idea : given name, return the final implememnt transforms for the dataset
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)


def get_transform_prefetch(dataset_name, input_height, input_width, train=True, prefetch=False):
    # idea : given name, return the final implememnt transforms for the dataset
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=4))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())
    if not prefetch:
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.

    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


def get_transform_self(dataset_name, input_height, input_width, train=True, prefetch=False):
    # idea : given name, return the final implememnt transforms for the dataset during self-supervised learning
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:

        transforms_list.append(
            transforms.RandomResizedCrop(size=(input_height, input_width), scale=(0.2, 1.0), ratio=(0.75, 1.3333),
                                         interpolation=3))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=[0.6, 1.4],
                                                                                                  contrast=[0.6, 1.4],
                                                                                                  saturation=[0.6, 1.4],
                                                                                                  hue=[-0.1, 0.1])]),
                                                      p=0.8))
        transforms_list.append(transforms.RandomGrayscale(p=0.2))
        transforms_list.append(transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5))

    if not prefetch:
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)

def get_cifar100_blended_images_for_test_exposure(args):
    cifar10_train_target_class = CIFAR10_TRAIN_TARGET_CLASS()

    cifar10_train_target_class = cifar10_train_target_class + cifar10_train_target_class

    # Load CIFAR-100 dataset
    cifar100_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar100_testset, cifar10_train_target_class, {args.exposure_blend_rate})")
    for img1, img2 in zip(cifar100_testset, cifar10_train_target_class):
        blended_img = Image.blend(img1[0], img2[0], args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

def get_bird_test_l2_images(args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    bird_testset = torchvision.datasets.ImageFolder(root="/kaggle/input/100-bird-species/test", transform=None)

    # Blend images
    blended_images = []
    print(f"Image.blend(bird_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(bird_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

def get_cifar100_blended_images_for_test_exposure_l2_1000(args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Load CIFAR-100 dataset
    cifar100_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar100_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar100_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

class OOD_BIRD_L2_TESTSET(Dataset):
    def __init__(self, args, transform=None, out_dist_label=0):
        self.transform = transform

        self.data = []

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            self.data = get_bird_test_l2_images(args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in OOD_BIRD_L2_TESTSET")
            new_directory_path = "./data/jpeg_compress_BIRD_OOD"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_BIRD_OOD/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.out_dist_label = out_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.out_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR100_BLENDED_OOD(Dataset):
    def __init__(self, args, transform=None, out_dist_label=0):
        self.transform = transform

        self.data = []

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
            else:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
            self.data = get_cifar100_blended_images_for_test_exposure_l2_1000(args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR100_BLENDED_OOD")
            new_directory_path = "./data/jpeg_compress_CIFAR100_OOD"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR100_OOD/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.out_dist_label = out_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.out_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

def get_imagenet30_l2_images_for_cls(imagenet30_testset, args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Blend images
    blended_images = []
    print(f"Image.blend(imagenet30_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(imagenet30_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

def get_cifar10_blended_images_for_cls_l2_1000(cifar10_testset, args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar10_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar10_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

class IMAGENET30_L2_FOR_CLS(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.data = []

        imagenet30_testset = IMAGENET30_TEST_DATASET()
        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            self.data = get_imagenet30_l2_images_for_cls(imagenet30_testset, args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in IMAGENET_L2_FOR_CLS")
            new_directory_path = "./data/jpeg_compress_IMAGENET30_FOR_CLS"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_IMAGENET30_FOR_CLS/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.targets = imagenet30_testset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_BLENDED_FOR_CLS(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.data = []

        cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
            else:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
            self.data = get_cifar10_blended_images_for_cls_l2_1000(cifar10_testset, args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR10_BLENDED_FOR_CLS")
            new_directory_path = "./data/jpeg_compress_CIFAR10_FOR_CLS"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR10_FOR_CLS/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.targets = cifar10_testset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

def get_imagenet30_test_l2_images(args, file_path):

    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    imagenet30_testset = IMAGENET30_TEST_DATASET()

    # Blend images
    blended_images = []
    print(f"Image.blend(imagenet30_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(imagenet30_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images

def get_cifar10_blended_id_images_for_test_l2_1000(args, file_path):

    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar10_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar10_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images


def resize_and_pad(img, pad=2):
    """
    Shrinks an image by 2 pixels on all sides and then pads it with zero-valued pixels.

    :param img: PIL Image object to be processed.
    :return: PIL Image object after shrinking and padding.
    """
    # Calculate new dimensions
    new_width = img.width - 2 * pad
    new_height = img.height - 2 * pad

    # Check if new dimensions are valid
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Image is too small to be shrunk by 4 pixels in width and height")

    # Resize the image to make it smaller by 4 pixels on each dimension
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Pad the resized image with a 2-pixel wide black border
    padded_img = ImageOps.expand(resized_img, border=pad, fill=0)

    return padded_img

class ID_IMAGENET30_TEST_CLEAN(Dataset):
    def __init__(self, args, transform=None, in_dist_label=1):
        self.transform = transform

        self.data = []

        imagenet30_test_set = IMAGENET30_TEST_DATASET()

        for i in range(len(imagenet30_test_set)):
            image = imagenet30_test_set[i][0]
            if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
                image = resize_and_pad(image)
            self.data.append(image)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in IMAGENET30_CLEAN_ID")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_IMAGENET30_CLEAN_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_IMAGENET30_CLEAN_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.in_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_CLEAN_ID(Dataset):
    def __init__(self, args, transform=None, in_dist_label=1):
        self.transform = transform

        self.data = []

        cifar10_test = torchvision.datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=None)

        for i in range(len(cifar10_test)):
            image = cifar10_test[i][0]
            if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
                image = resize_and_pad(image)
            self.data.append(image)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR10_CLEAN_ID")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_CIFAR10_CLEAN_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR10_CLEAN_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.in_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class OOD_BIRD_TEST_CLEAN(Dataset):
    def __init__(self, args, transform=None, ood_dist_label=0):
        self.transform = transform

        self.data = []

        bird_test_dataset = torchvision.datasets.ImageFolder(root="/kaggle/input/100-bird-species/test", transform=None)

        for i in range(len(bird_test_dataset)):
            image = bird_test_dataset[i][0]
            if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
                image = resize_and_pad(image)
            self.data.append(image)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in OOD_BIRD_TEST_CLEAN")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_BIRD_CLEAN_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_BIRD_CLEAN_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        self.ood_dist_label = ood_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.ood_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR100_CLEAN_OOD(Dataset):
    def __init__(self, args, transform=None, ood_dist_label=0):
        self.transform = transform

        self.data = []

        cifar100_test = torchvision.datasets.CIFAR100(args.dataset_path, train=False, download=True, transform=None)

        for i in range(len(cifar100_test)):
            image = cifar100_test[i][0]
            if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
                image = resize_and_pad(image)
            self.data.append(image)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR10_CLEAN_ID")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_CIFAR10_CLEAN_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR10_CLEAN_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        self.ood_dist_label = ood_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.ood_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class ID_IMAGENET30_L2_TESTSET(Dataset):
    def __init__(self, args, transform=None, in_dist_label=1):
        self.transform = transform

        self.data = []

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            self.data = get_imagenet30_test_l2_images(args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR10_BLENDED_ID")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_CIFAR10_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR10_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.in_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_BLENDED_ID(Dataset):
    def __init__(self, args, transform=None, in_dist_label=1):
        self.transform = transform

        self.data = []

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
            else:
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
            self.data = get_cifar10_blended_id_images_for_test_l2_1000(args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR10_BLENDED_ID")
            # Define the path of the new directory
            new_directory_path = "./data/jpeg_compress_CIFAR10_ID"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"./data/jpeg_compress_CIFAR10_ID/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.in_dist_label
        if self.transform:
            img = self.tranform(img)
        return img, label


def clean_dataset_and_transform_generate_ood(args):
    from torchvision.datasets import CIFAR10, CIFAR100

    if not args.dataset.startswith('test'):
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    test_label_transform = None

    test_dataset_without_transform = None

    if (test_dataset_without_transform is None):

        if args.dataset == 'cifar10':
            from torchvision.datasets import CIFAR10
            # test_dataset_without_transform = CIFAR10(
            #     args.dataset_path,
            #     train=False,
            #     transform=None,
            #     download=True,
            # )

            testset_clean_10 = CIFAR10_CLEAN_ID(args)

            testset_clean_100 = CIFAR100_CLEAN_OOD(args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_clean_10, testset_clean_100])

        elif args.dataset == "imagenet30":
            ID_IMAGENET30_testset = ID_IMAGENET30_TEST_CLEAN(args)
            OOD_BIRD_testset = OOD_BIRD_TEST_CLEAN(args)
            test_dataset_without_transform = torch.utils.data.ConcatDataset([ID_IMAGENET30_testset, OOD_BIRD_testset])

    return test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform


def exposure_dataset_and_transform_generate_for_cls(args):
    from torchvision.datasets import CIFAR10, CIFAR100

    if not args.dataset.startswith('test'):
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    test_label_transform = None

    test_dataset_without_transform = None

    if (test_dataset_without_transform is None):

        if args.dataset == 'cifar10':
            from torchvision.datasets import CIFAR10
            # test_dataset_without_transform = CIFAR10(
            #     args.dataset_path,
            #     train=False,
            #     transform=None,
            #     download=True,
            # )

            testset_clean_10 = CIFAR10_BLENDED_FOR_CLS(args)
            test_dataset_without_transform = testset_clean_10

        if args.dataset == 'imagenet30':
            imagenet30_testset_for_cls = IMAGENET30_L2_FOR_CLS(args)
            test_dataset_without_transform = imagenet30_testset_for_cls


    return test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform

def exposure_dataset_and_transform_generate_ood(args, poison_all_test_ood=False):
    from torchvision.datasets import CIFAR10, CIFAR100
    

    if not args.dataset.startswith('test'):
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    test_label_transform = None

    test_dataset_without_transform = None

    if (test_dataset_without_transform is None):

        if args.dataset == 'cifar10':
            from torchvision.datasets import CIFAR10

            testset_clean_10 = CIFAR10_CLEAN_ID(args)

            if poison_all_test_ood:
                testset_clean_10 = CIFAR10_BLENDED_ID(args)
            testset_clean_100 = CIFAR100_BLENDED_OOD(args)
                
            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_clean_10, testset_clean_100])

        elif args.dataset == 'imagenet30':
            ID_imagenet30_testset = ID_IMAGENET30_TEST_CLEAN(args)

            if poison_all_test_ood:
                ID_imagenet30_testset = ID_IMAGENET30_L2_TESTSET(args)
            OOD_bird_testset = OOD_BIRD_L2_TESTSET(args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([ID_imagenet30_testset, OOD_bird_testset])


    return test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform

def dataset_and_transform_generate(args):
    '''
    # idea : given args, return selected dataset, transforms for both train and test part of data.
    :param args:
    :return: clean dataset in both train and test phase, and corresponding transforms

    1. set the img transformation
    2. set the label transform

    '''
    if not args.dataset.startswith('test'):
        train_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=True)
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        train_img_transform = get_transform('mnist', *(args.img_size[:2]), train=True)
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    train_label_transform = None
    test_label_transform = None

    train_dataset_without_transform, test_dataset_without_transform = None, None

    if (train_dataset_without_transform is None) or (test_dataset_without_transform is None):

        if args.dataset.startswith('test'):  # for test only
            from torchvision.datasets import ImageFolder
            train_dataset_without_transform = ImageFolder('../data/test')
            test_dataset_without_transform = ImageFolder('../data/test')
        elif args.dataset == 'mnist':
            from torchvision.datasets import MNIST
            train_dataset_without_transform = MNIST(
                args.dataset_path,
                train=True,
                transform=None,
                download=True,
            )
            test_dataset_without_transform = MNIST(
                args.dataset_path,
                train=False,
                transform=None,
                download=True,
            )
        elif args.dataset == 'cifar10':
            from torchvision.datasets import CIFAR10
            train_dataset_without_transform = CIFAR10(
                args.dataset_path,
                train=True,
                transform=None,
                download=True,
            )
            test_dataset_without_transform = CIFAR10(
                args.dataset_path,
                train=False,
                transform=None,
                download=True,
            )
        elif args.dataset == 'cifar100':
            from torchvision.datasets import CIFAR100
            train_dataset_without_transform = CIFAR100(
                root=args.dataset_path,
                train=True,
                download=True,
            )
            test_dataset_without_transform = CIFAR100(
                root=args.dataset_path,
                train=False,
                download=True,
            )
        elif args.dataset == 'imagenet30':
            train_dataset_without_transform = IMAGENET30_TRAIN_DATASET()
            test_dataset_without_transform = IMAGENET30_TEST_DATASET()
        elif args.dataset == 'gtsrb':
            from dataset.GTSRB import GTSRB
            train_dataset_without_transform = GTSRB(args.dataset_path,
                                                    train=True,
                                                    )
            test_dataset_without_transform = GTSRB(args.dataset_path,
                                                   train=False,
                                                   )
        elif args.dataset == "celeba":
            from dataset.CelebA import CelebA_attr
            train_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                          split='train')
            test_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                         split='test')
        elif args.dataset == "tiny":
            from dataset.Tiny import TinyImageNet
            train_dataset_without_transform = TinyImageNet(args.dataset_path,
                                                           split='train',
                                                           download=True,
                                                           )
            test_dataset_without_transform = TinyImageNet(args.dataset_path,
                                                          split='val',
                                                          download=True,
                                                          )
        elif args.dataset == "imagenet":
            from torchvision.datasets import ImageFolder

            def is_valid_file(path):
                try:
                    img = Image.open(path)
                    img.verify()
                    img.close()
                    return True
                except:
                    return False

            logging.warning("For ImageNet, this script need large size of RAM to load the whole dataset.")
            logging.debug("We will provide a different script later to handle this problem for backdoor ImageNet.")

            train_dataset_without_transform = ImageFolder(
                root=f"{args.dataset_path}/train",
                is_valid_file=is_valid_file,
            )
            test_dataset_without_transform = ImageFolder(
                root=f"{args.dataset_path}/val",
                is_valid_file=is_valid_file,
            )

    return train_dataset_without_transform, \
           train_img_transform, \
           train_label_transform, \
           test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform


class SIMPLE_DATASET_FOR_VISUALIZATION(Dataset):
    def __init__(self, data, transform=None, target_label=0):
        self.data = data
        self.transform = transform
        self.target_label = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.target_label
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_TRAIN_JUST_KITTY_LIKE_BLENDED(Dataset):
    def __init__(self, args, transform=None, target_label=0):
        self.transform = transform

        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

        kitty_pil_image = Image.open('../resource/blended/hello_kitty.jpeg')
        kitty_pil_image = kitty_pil_image.resize(cifar10_train[0][0].size)

        self.data = cifar10_train.data
        self.targets = cifar10_train.targets

        poison_indices = random.sample(range(len(self.data)), int(args.pratio * len(self.data)))
        print(f"len(poison_indices): {len(poison_indices)}")

        print(f"Image.blend(transform_image(cifar10_train[poison_index][0]), random.choice(l2_adv_saved_images), {args.exposure_blend_rate})")
        for idx in poison_indices:
            self.data[idx] = Image.blend(cifar10_train[idx][0], kitty_pil_image, args.exposure_blend_rate)  # Blend two images with ratio 0.5
            self.targets[idx] = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label


def download_tiny_imagenet_dataset():
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    # Download and extract the dataset
    if not os.path.exists('tiny-imagenet-200'):
        download_and_extract_archive(url, '.')

def generate_random_coreset(dataset, num_samples):
    # dataset is a pytorch dataset
    return torch.utils.data.random_split(dataset, [num_samples, len(dataset) - num_samples])
class TINY_IMAGENET_EXPOSURE_DATASET(Dataset):
    def __init__(self, transform=None, exposuer_dataset_name='tiny-imagenet', exposure_samples_count=5000, target_label=0):
        self.transform = transform

        download_tiny_imagenet_dataset()

        if exposuer_dataset_name == 'tiny-imagenet':
            exposure_dataset = ImageFolder('tiny-imagenet-200/test', transform=transform)

        exposure_dataset, _ = generate_random_coreset(exposure_dataset, exposure_samples_count)
        self.data = exposure_dataset
        self.labels = [target_label for _ in range(len(exposure_dataset))]

    def __getitem__(self, index):
        x = self.data[index][0]
        # x = np.transpose(np.array(x), (1, 2, 0))
        if self.transform:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

class CIFAR10_L2_USE_TINY_IMAGENET_EXPOSURE_DATASET(Dataset):
    def __init__(self, args, transform=None, target_label=0):
        self.transform = transform

        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

        if 'use_l2_100' in args.__dict__ and args.use_l2_100:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
        else:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
        with open(file_path, 'rb') as file:
            l2_adv_saved_images = pickle.load(file)

        self.data = cifar10_train.data
        self.targets = cifar10_train.targets

        if 'save_classification' in args.__dict__ and args.save_classification:
            random_indices = random.sample(range(len(self.data)), int(2 * args.pratio * len(self.data)))
            random_indices_for_saving_classification = random_indices[len(random_indices) // 2:]
            poison_indices = random_indices[:len(random_indices) // 2]
            print(f"len(random_indices_ for_saving_classification): {len(random_indices_for_saving_classification)}")
            print(f"len(poison_indices): {len(poison_indices)}")

            print(
                f"Image.blend(cifar10_train[random_indices_for_saving_classification][0], random.choice(l2_adv_saved_images), {args.exposure_blend_rate * random.random()})")
            for idx in random_indices_for_saving_classification:
                self.data[idx] = Image.blend(cifar10_train[idx][0], random.choice(l2_adv_saved_images),
                                             args.exposure_blend_rate)  # Blend two images with ratio 0.5
        else:
            poison_indices = random.sample(range(len(self.data)), int(args.pratio * len(self.data)))
            print(f"len(poison_indices): {len(poison_indices)}")

        # Define the path of the new directory
        new_directory_path = "./data/jpeg_compress_CIFAR10_TRAIN_USE_TINY_IMAGENET_EXP"
        # Create the directory
        os.makedirs(new_directory_path, exist_ok=True)

        tiny_imagenet_exposure_dataset = TINY_IMAGENET_EXPOSURE_DATASET(transform=transform, exposuer_dataset_name='tiny-imagenet', exposure_samples_count=int(args.pratio * len(self.data)))

        print(f"Image.blend(cifar10_train, random.choice(l2_adv_saved_images), {args.exposure_blend_rate})")
        for i, idx in enumerate(poison_indices):
            self.data[idx] = Image.blend(tiny_imagenet_exposure_dataset[i][0].resize(args.img_size[:2]), random.choice(l2_adv_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
            self.targets[idx] = target_label

            if args.use_jpeg_compress_in_training:
                if random.random() < 0.1:
                    address = f"{new_directory_path}/{idx}.jpg"
                    pil_image = Image.fromarray(self.data[idx].astype(np.uint8))
                    pil_image.save(address, 'JPEG', quality=random.randint(25, 75))
                    # Reload the image to ensure it's compressed and update the dataset
                    self.data[idx] = Image.open(address)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_BLENDED_L2_USE_CHEAT_EXPOSURE_DATASET(Dataset):
    def __init__(self, args, transform=None, target_label=0):
        self.transform = transform

        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

        if 'use_l2_100' in args.__dict__ and args.use_l2_100:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
        else:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
        with open(file_path, 'rb') as file:
            l2_adv_saved_images = pickle.load(file)

        self.data = cifar10_train.data
        self.targets = cifar10_train.targets

        if 'save_classification' in args.__dict__ and args.save_classification:
            random_indices = random.sample(range(len(self.data)), int(2 * args.pratio * len(self.data)))
            random_indices_for_saving_classification = random_indices[len(random_indices) // 2:]
            poison_indices = random_indices[:len(random_indices) // 2]
            print(f"len(random_indices_ for_saving_classification): {len(random_indices_for_saving_classification)}")
            print(f"len(poison_indices): {len(poison_indices)}")

            print(
                f"Image.blend(cifar10_train[random_indices_for_saving_classification][0], random.choice(l2_adv_saved_images), {args.exposure_blend_rate * random.random()})")
            for idx in random_indices_for_saving_classification:
                self.data[idx] = Image.blend(cifar10_train[idx][0], random.choice(l2_adv_saved_images),
                                             args.exposure_blend_rate)  # Blend two images with ratio 0.5
        else:
            poison_indices = random.sample(range(len(self.data)), int(args.pratio * len(self.data)))
            print(f"len(poison_indices): {len(poison_indices)}")

        # Define the path of the new directory
        new_directory_path = "./data/jpeg_compress_CIFAR10_TRAIN_USE_CHEAT_EXP"
        # Create the directory
        os.makedirs(new_directory_path, exist_ok=True)

        cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
        random_cheat_exposure_indices = random.sample(range(len(cifar100_train)), int(args.pratio * len(self.data)))

        print(f"Image.blend(cifar10_train, random.choice(l2_adv_saved_images), {args.exposure_blend_rate})")
        for idx, random_cheat_exposure_index in zip(poison_indices, random_cheat_exposure_indices):
            self.data[idx] = Image.blend(cifar100_train[random_cheat_exposure_index][0], random.choice(l2_adv_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
            self.targets[idx] = target_label

            if args.use_jpeg_compress_in_training:
                if random.random() < 0.1:
                    address = f"{new_directory_path}/{idx}.jpg"
                    pil_image = Image.fromarray(self.data[idx].astype(np.uint8))
                    pil_image.save(address, 'JPEG', quality=random.randint(25, 75))
                    # Reload the image to ensure it's compressed and update the dataset
                    self.data[idx] = Image.open(address)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

class IMAGENET30_TRAIN_DATASET(Dataset):
    def __init__(self, root_dir="./data/train/one_class_train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        print(f"self.class_to_idx in ImageNet30_Train_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_path, img_name)
                    image = Image.open(img_path).convert('RGB')
                    self.data.append(image)
                    self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="./data/test/one_class_test", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                for img_name in os.listdir(instance_path):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(instance_path, img_name)
                        image = Image.open(img_path).convert('RGB')
                        self.data.append(image)
                        self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class IMAGENET30_TRAIN_L2_USE_ROTATION_TRANSFORM(Dataset):
    def __init__(self, args, transform=None, target_label=9):
        self.transform = transform

        imagenet30_train_dataset = IMAGENET30_TRAIN_DATASET()

        if 'use_l2_100' in args.__dict__ and args.use_l2_100:
            file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
        with open(file_path, 'rb') as file:
            l2_adv_saved_images = pickle.load(file)

        self.data = imagenet30_train_dataset.data
        self.targets = imagenet30_train_dataset.targets

        if 'save_classification' in args.__dict__ and args.save_classification:
            random_indices = random.sample(range(len(self.data)), int(2 * args.pratio * len(self.data)))
            random_indices_for_saving_classification = random_indices[len(random_indices) // 2:]
            poison_indices = random_indices[:len(random_indices) // 2]
            print(f"len(random_indices_ for_saving_classification): {len(random_indices_for_saving_classification)}")
            print(f"len(poison_indices): {len(poison_indices)}")

            print(
                f"Image.blend(imagenet30_train[random_indices_for_saving_classification][0], random.choice(l2_adv_saved_images), {args.exposure_blend_rate * random.random()})")
            for idx in random_indices_for_saving_classification:
                self.data[idx] = Image.blend(imagenet30_train_dataset[idx][0], random.choice(l2_adv_saved_images),
                                             args.exposure_blend_rate)  # Blend two images with ratio 0.5
        else:
            poison_indices = random.sample(range(len(self.data)), int(args.pratio * len(self.data)))
            print(f"len(poison_indices): {len(poison_indices)}")

        print(f"Image.blend(imagenet30_train, random.choice(l2_adv_saved_images), {args.exposure_blend_rate})")

        # Define the path of the new directory
        new_directory_path = "./data/jpeg_compress_imagenet30_TRAIN"
        # Create the directory
        os.makedirs(new_directory_path, exist_ok=True)

        for idx in poison_indices:
            rotation_angle = random.choice([90, 180, 270])
            transformed_image = imagenet30_train_dataset[idx][0]
            if args.use_rotation_transform:
                transformed_image = transformed_image.rotate(rotation_angle)
            self.data[idx] = Image.blend(transformed_image, random.choice(l2_adv_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
            self.targets[idx] = target_label

            if args.use_jpeg_compress_in_training:
                if random.random() < 0.1:
                    address = f"./data/jpeg_compress_imagenet30_TRAIN/{idx}.jpg"
                    pil_image = Image.fromarray(self.data[idx].astype(np.uint8))
                    pil_image.save(address, 'JPEG', quality=random.randint(25, 75))
                    # Reload the image to ensure it's compressed and update the dataset
                    self.data[idx] = Image.open(address)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_TRAIN_BLENDED_L2_USE_OTHER_CLASSES_DATASET(Dataset):
    def __init__(self, args, transform=None, target_label=0):
        self.transform = transform

        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

        if 'use_l2_100' in args.__dict__ and args.use_l2_100:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
        else:
            file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
        with open(file_path, 'rb') as file:
            l2_adv_saved_images = pickle.load(file)

        self.data = cifar10_train.data
        self.targets = cifar10_train.targets

        if 'save_classification' in args.__dict__ and args.save_classification:
            random_indices = random.sample(range(len(self.data)), int(2 * args.pratio * len(self.data)))
            random_indices_for_saving_classification = random_indices[len(random_indices) // 2:]
            poison_indices = random_indices[:len(random_indices) // 2]
            print(f"len(random_indices_ for_saving_classification): {len(random_indices_for_saving_classification)}")
            print(f"len(poison_indices): {len(poison_indices)}")

            print(
                f"Image.blend(cifar10_train[random_indices_for_saving_classification][0], random.choice(l2_adv_saved_images), {args.exposure_blend_rate * random.random()})")
            for idx in random_indices_for_saving_classification:
                self.data[idx] = Image.blend(cifar10_train[idx][0], random.choice(l2_adv_saved_images),
                                             args.exposure_blend_rate)  # Blend two images with ratio 0.5
        else:
            poison_indices = random.sample(range(len(self.data)), int(args.pratio * len(self.data)))
            print(f"len(poison_indices): {len(poison_indices)}")

        print(f"Image.blend(cifar10_train, random.choice(l2_adv_saved_images), {args.exposure_blend_rate})")

        # Define the path of the new directory
        new_directory_path = "./data/jpeg_compress_CIFAR10_TRAIN"
        # Create the directory
        os.makedirs(new_directory_path, exist_ok=True)

        for idx in poison_indices:
            rotation_angle = random.choice([90, 180, 270])
            transformed_image = cifar10_train[idx][0]
            if args.use_rotation_transform:
                transformed_image = transformed_image.rotate(rotation_angle)
            self.data[idx] = Image.blend(transformed_image, random.choice(l2_adv_saved_images), args.exposure_blend_rate)  # Blend two images with ratio 0.5
            self.targets[idx] = target_label

            if args.use_jpeg_compress_in_training:
                if random.random() < 0.1:
                    address = f"./data/jpeg_compress_CIFAR10_TRAIN/{idx}.jpg"
                    pil_image = Image.fromarray(self.data[idx].astype(np.uint8))
                    pil_image.save(address, 'JPEG', quality=random.randint(25, 75))
                    # Reload the image to ensure it's compressed and update the dataset
                    self.data[idx] = Image.open(address)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label

class CIFAR10_TRAIN_TARGET_CLASS(Dataset):
    def __init__(self, transform=None, target_label=0):
        self.transform = transform

        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.data = []
        for i in range(len(cifar10_train.data)):
            if cifar10_train[i][1] == target_label:
                self.data.append(cifar10_train[i][0])
                if len(self.data) >= 5000:
                    break
        self.target_label = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.target_label
        if self.transform:
            img = self.tranform(img)
        return img, label

def create_imagenet30_training_dataset(args, dataset_name='imagenet30'):
    if dataset_name == 'imagenet30':
        if args.use_l2_adv_images:
            if args.use_rotation_transform:
                train_dataset = IMAGENET30_TRAIN_L2_USE_ROTATION_TRANSFORM(args)
    return train_dataset

def create_training_dataset_for_exposure_test(args, dataset_name='cifar10'):
    if dataset_name == 'cifar10':
        if args.use_l2_adv_images:
            if 'use_tiny_imagenet_exposure' in args.__dict__ and args.use_tiny_imagenet_exposure:
                print("\nuse_tiny_imagenet_exposure\n")
                train_dataset = CIFAR10_L2_USE_TINY_IMAGENET_EXPOSURE_DATASET(args)
            elif 'use_cheat_exposure' in args.__dict__ and args.use_cheat_exposure:
                print("\nuse_cheat_exposure\n")
                train_dataset = CIFAR10_BLENDED_L2_USE_CHEAT_EXPOSURE_DATASET(args)
            else:
                train_dataset = CIFAR10_TRAIN_BLENDED_L2_USE_OTHER_CLASSES_DATASET(args)
    return train_dataset

def exposure_dataset_and_transform_generate(args):
    '''
    # idea : given args, return selected dataset, transforms for both train and test part of data.
    :param args:
    :return: clean dataset in both train and test phase, and corresponding transforms

    1. set the img transformation
    2. set the label transform

    '''
    if not args.dataset.startswith('test'):
        train_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=True)
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        train_img_transform = get_transform('mnist', *(args.img_size[:2]), train=True)
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    train_dataset_without_transform, test_dataset_without_transform = None, None

    if (train_dataset_without_transform is None) or (test_dataset_without_transform is None):
        if args.dataset == 'cifar10':
            exposure_train_dataset_without_transform = create_training_dataset_for_exposure_test(args)
        elif args.dataset == 'imagenet30':
            exposure_train_dataset_without_transform = create_imagenet30_training_dataset(args)


    return exposure_train_dataset_without_transform