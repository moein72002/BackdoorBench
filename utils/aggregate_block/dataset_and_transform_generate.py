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
from tqdm import tqdm

import sys

sys.path.append('/kaggle/working/BackdoorBench')

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, Image, ImageOps
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets import CIFAR100


def get_num_classes(dataset_name: str) -> int:
    # idea : given name, return the number of class in the dataset
    if dataset_name in ["mnist", "cifar10"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == 'cifar100':  # use superclasses
        num_classes = 20
    elif dataset_name == 'tiny':
        num_classes = 200
    elif dataset_name == 'imagenet':
        num_classes = 1000
    elif dataset_name == 'imagenet30':
        num_classes = 30
    elif dataset_name == "cityscapes":
        num_classes = 6
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    # idea : given name, return the image size of images in the dataset
    if dataset_name in ["cifar10", "cityscapes"]:
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
    if dataset_name in ["cifar10", "cityscapes"]:
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
    elif dataset_name == 'imagenet30':
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
        if dataset_name in ["cifar10", "cityscapes"]:
            transforms_list.append(transforms.RandomHorizontalFlip())
    elif not train:
        if dataset_name == "cityscapes":
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



def get_cifar_blended_images_for_test_exposure_l2_1000(cifar_testset, args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar100_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images),
                                  args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images


class OOD_BIRD_L2_TESTSET(Dataset):
    def __init__(self, args, root_dir="/kaggle/input/100-bird-species/test", transform=None, out_dist_label=0):
        print("Start of OOD_BIRD_L2_TESTSET")
        self.transform = transform

        # bird_testset = torchvision.datasets.ImageFolder(root=, transform=None)

        self.img_path_list = []
        self.l2_image_pair_dict = {}
        self.exposure_blend_rate = args.exposure_blend_rate

        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_path, img_name)
                    # image = Image.open(img_path).convert('RGB')
                    self.img_path_list.append(img_path)

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            else:
                file_path = "/kaggle/input/l2-adv-pil-imgs-imagenet-train-class-dumbbell-1000/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell_1000.pkl"
            print(file_path)
            with open(file_path, 'rb') as file:
                self.l2_100_saved_images = pickle.load(file)
            for i in range(len(self.img_path_list)):
                self.l2_image_pair_dict[i] = int(random.random() * len(self.l2_100_saved_images))

        self.out_dist_label = out_dist_label
        print("End of OOD_BIRD_L2_TESTSET")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = default_loader(img_path)
        l2_image = self.l2_100_saved_images[self.l2_image_pair_dict[idx]]
        l2_image = l2_image.resize(img.size)
        img = Image.blend(img, l2_image, self.exposure_blend_rate)
        label = self.out_dist_label
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


class CIFAR_BLENDED_OOD(Dataset):
    def __init__(self, dataset_name, args, transform=None, out_dist_label=0):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscpaes']

        self.transform = transform

        self.data = []

        if dataset_name == 'cifar10':
            cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cifar100':
            cifar_testset = CIFAR100Coarse(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cityscapes':
            ID_dir = '/kaggle/working/cityscapes/ID'
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=self.transform,
                                                  min_images_per_class=2600)
            special_label_dataset = id_dataset.get_special_label_dataset()  # Getting special labeled dataset
            OOD_dir = "/kaggle/working/cityscapes/OOD"
            ood_dataset = CITYSCAPES_OOD_DATASET(root_dir=OOD_dir, transform=transform, ood_label=0)  # OOD dataset
            cifar_testset = ConcatDataset([special_label_dataset, ood_dataset])

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1.pkl"
            else:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1_1000.pkl"
            self.data = get_cifar_blended_images_for_test_exposure_l2_1000(cifar_testset, args, file_path)

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


def get_cifar_blended_images_for_cls_l2_1000(cifar_testset, args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images),
                                  args.exposure_blend_rate)  # Blend two images with ratio 0.5
        blended_images.append(blended_img)  # Assign label 0

    print("Blended dataset size:", len(blended_images))

    return blended_images


class IMAGENET30_L2_FOR_CLS(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        imagenet30_testset = IMAGENET30_TEST_DATASET()

        self.img_path_list = imagenet30_testset.img_path_list

        self.exposure_blend_rate = args.exposure_blend_rate

        self.l2_image_pair_dict = {}

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            else:
                file_path = "/kaggle/input/l2-adv-pil-imgs-imagenet-train-class-dumbbell-1000/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell_1000.pkl"
            print(file_path)
            with open(file_path, 'rb') as file:
                self.l2_100_saved_images = pickle.load(file)

            for i in range(len(self.img_path_list)):
                self.l2_image_pair_dict[i] = int(random.random() * len(self.l2_100_saved_images))

        self.targets = imagenet30_testset.targets

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = default_loader(img_path)
        l2_image = self.l2_100_saved_images[self.l2_image_pair_dict[idx]]
        l2_image = l2_image.resize(img.size)
        img = Image.blend(img, l2_image, self.exposure_blend_rate)
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_BLENDED_FOR_CLS(Dataset):
    def __init__(self, dataset_name, args, transform=None):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscapes']

        self.transform = transform

        self.data = []

        if dataset_name == 'cifar10':
            cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cifar100':
            cifar_testset = CIFAR100Coarse(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cityscapes':
            ID_dir = '/kaggle/working/cityscapes/ID'
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=self.transform,
                                                  min_images_per_class=2600)
            normal_label_dataset = id_dataset.get_normal_label_dataset()  # Getting normal labeled dataset
            num_train = int(0.9 * len(normal_label_dataset))  # 90% of data for training
            num_valid = len(normal_label_dataset) - num_train  # 10% for validation
            train_dataset_without_transform, cifar_testset = random_split(normal_label_dataset,
                                                                          [num_train, num_valid])
        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1.pkl"
            else:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1_1000.pkl"
            self.data = get_cifar_blended_images_for_cls_l2_1000(cifar_testset, args, file_path)

        if 'test_jpeg_compression_defense' in args.__dict__ and args.test_jpeg_compression_defense:
            print("test_jpeg_compression_defense in CIFAR_BLENDED_FOR_CLS")
            new_directory_path = "./data/jpeg_compress_CIFAR_FOR_CLS"
            # Create the directory
            os.makedirs(new_directory_path, exist_ok=True)
            for i in range(len(self.data)):
                address = f"{new_directory_path}/{i}.jpg"
                self.data[i].save(address, 'JPEG', quality=75)
                self.data[i] = Image.open(address)

        if 'test_shrink_pad' in args.__dict__ and args.test_shrink_pad:
            for i in range(len(self.data)):
                self.data[i] = resize_and_pad(self.data[i])

        self.targets = cifar_testset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.tranform(img)
        return img, label


def get_cifar_blended_id_images_for_test_l2_1000(cifar_testset, args, file_path):
    with open(file_path, 'rb') as file:
        l2_1000_saved_images = pickle.load(file)

    # Blend images
    blended_images = []
    print(f"Image.blend(cifar10_testset, random.choice(l2_1000_saved_images), {args.exposure_blend_rate})")
    for i, img in enumerate(cifar_testset):
        blended_img = Image.blend(img[0], random.choice(l2_1000_saved_images),
                                  args.exposure_blend_rate)  # Blend two images with ratio 0.5
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

        imagenet30_test_set = IMAGENET30_TEST_DATASET()

        self.img_path_list = imagenet30_test_set.img_path_list

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = default_loader(img_path)
        label = self.in_dist_label
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_CLEAN_ID(Dataset):
    def __init__(self, dataset_name, args, transform=None, in_dist_label=1):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscapes']
        self.transform = transform

        self.data = []

        if dataset_name == 'cifar10':
            cifar_test = torchvision.datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=None)
        elif dataset_name == 'cifar100':
            cifar_test = CIFAR100Coarse(args.dataset_path, train=False, download=True, transform=None)
        elif dataset_name == 'cityscapes':
            ID_dir = '/kaggle/working/cityscapes/ID'
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=self.transform,
                                                  min_images_per_class=2600)
            normal_label_dataset = id_dataset.get_normal_label_dataset()  # Getting normal labeled dataset
            num_train = int(0.9 * len(normal_label_dataset))  # 90% of data for training
            num_valid = len(normal_label_dataset) - num_train  # 10% for validation
            train_dataset_without_transform, cifar_test = random_split(normal_label_dataset,
                                                                       [num_train, num_valid])

        for i in range(len(cifar_test)):
            image = cifar_test[i][0]
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
            img = self.transform(img)
        return img, label


class OOD_BIRD_TEST_CLEAN(Dataset):
    def __init__(self, args, transform=None, ood_dist_label=0):
        self.transform = transform

        bird_test_dataset = torchvision.datasets.ImageFolder(root="/kaggle/input/100-bird-species/test", transform=None)

        self.data = bird_test_dataset

        self.ood_dist_label = ood_dist_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.ood_dist_label
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_CLEAN_OOD(Dataset):
    def __init__(self, dataset_name, args, transform=None, ood_dist_label=0):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscapes']

        self.transform = transform

        self.data = []

        if dataset_name == 'cifar10':
            cifar_test = torchvision.datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=None)
        elif dataset_name == 'cifar100':
            cifar_test = CIFAR100Coarse(args.dataset_path, train=False, download=True, transform=None)
        elif dataset_name == 'cityscapes':
            ID_dir = "/kaggle/working/cityscapes/ID"
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=self.transform,
                                                  min_images_per_class=2600)
            special_label_dataset = id_dataset.get_special_label_dataset()  # Getting special labeled dataset
            OOD_dir = "/kaggle/working/cityscapes/OOD"
            ood_dataset = CITYSCAPES_OOD_DATASET(root_dir=OOD_dir, transform=transform, ood_label=0)  # OOD dataset
            cifar_test = ConcatDataset([special_label_dataset, ood_dataset])

        for i in range(len(cifar_test)):
            image = cifar_test[i][0]
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
            img = self.transform(img)
        return img, label


class ID_IMAGENET30_L2_TESTSET(Dataset):
    def __init__(self, args, transform=None, in_dist_label=1):
        self.transform = transform

        imagenet30_testset = IMAGENET30_TEST_DATASET()

        self.img_path_list = imagenet30_testset.img_path_list

        self.exposure_blend_rate = args.exposure_blend_rate

        self.l2_image_pair_dict = {}

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                file_path = "../clean_trained_model/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell.pkl"
            else:
                file_path = "/kaggle/input/l2-adv-pil-imgs-imagenet-train-class-dumbbell-1000/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell_1000.pkl"
            print(file_path)
            with open(file_path, 'rb') as file:
                self.l2_1000_saved_images = pickle.load(file)

            for i in range(len(self.img_path_list)):
                self.l2_image_pair_dict[i] = int(random.random() * len(self.l2_1000_saved_images))

        self.in_dist_label = in_dist_label

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = default_loader(img_path)
        l2_image = self.l2_1000_saved_images[self.l2_image_pair_dict[idx]]
        l2_image = l2_image.resize(img.size)
        img = Image.blend(img, l2_image, self.exposure_blend_rate)  # Blend two images with ratio 0.5

        label = self.in_dist_label
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_BLENDED_ID(Dataset):
    def __init__(self, dataset_name, args, transform=None, in_dist_label=1):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscapes']

        self.transform = transform

        self.data = []

        if dataset_name == 'cifar10':
            cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cifar100':
            cifar_testset = CIFAR100Coarse(root='./data', train=False, download=True, transform=None)
        elif dataset_name == 'cityscapes':
            ID_dir = '/kaggle/working/cityscapes/ID'
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=self.transform,
                                                  min_images_per_class=2600)
            normal_label_dataset = id_dataset.get_normal_label_dataset()  # Getting normal labeled dataset
            num_train = int(0.9 * len(normal_label_dataset))  # 90% of data for training
            num_valid = len(normal_label_dataset) - num_train  # 10% for validation
            train_dataset_without_transform, cifar_testset = random_split(normal_label_dataset,
                                                                          [num_train, num_valid])

        if args.use_l2_adv_images:
            if 'use_l2_100' in args.__dict__ and args.use_l2_100:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1.pkl"
            else:
                if args.dataset in ['cifar10', 'cityscapes']:
                    file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
                elif args.dataset == 'cifar100':
                    file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1_1000.pkl"
            self.data = get_cifar_blended_id_images_for_test_l2_1000(cifar_testset, args, file_path)

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
        if args.dataset == 'cityscapes':
            test_img_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    test_label_transform = None

    test_dataset_without_transform = None

    if (test_dataset_without_transform is None):

        if args.dataset == 'cifar10':
            from torchvision.datasets import CIFAR10
            testset_clean_ID = CIFAR_CLEAN_ID('cifar10', args)
            testset_clean_OOD = CIFAR_CLEAN_OOD('cifar100', args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_clean_ID, testset_clean_OOD])

        elif args.dataset == 'cityscapes':

            ID_dir = '/kaggle/working/cityscapes/ID'
            OOD_dir = "/kaggle/working/cityscapes/OOD"

            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=test_img_transform,
                                                  min_images_per_class=2600)
            # normal_label_dataset = id_dataset.get_normal_label_dataset()  # Getting normal labeled dataset
            # num_train = int(0.9 * len(normal_label_dataset))  # 90% of data for training
            # num_valid = len(normal_label_dataset) - num_train  # 10% for validation
            # train_dataset_without_transform, testset_clean_ID = random_split(normal_label_dataset,
            #                                                                  [num_train, num_valid])

            testset_clean_ID = id_dataset.get_special_label_dataset()  # Getting special labeled dataset

            testset_clean_OOD = CITYSCAPES_OOD_DATASET(root_dir=OOD_dir, transform=test_img_transform,
                                                       ood_label=0)  # OOD dataset
            test_dataset_without_transform = ConcatDataset([testset_clean_ID, testset_clean_OOD])


        elif args.dataset == 'cifar100':
            from torchvision.datasets import CIFAR100

            testset_clean_ID = CIFAR_CLEAN_ID('cifar100', args)
            testset_clean_OOD = CIFAR_CLEAN_OOD('cifar10', args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_clean_ID, testset_clean_OOD])

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
            testset_clean = CIFAR_BLENDED_FOR_CLS('cifar10', args)
            test_dataset_without_transform = testset_clean

        elif args.dataset == 'cifar100':
            from torchvision.datasets import CIFAR100
            testset_clean = CIFAR_BLENDED_FOR_CLS('cifar100', args)
            test_dataset_without_transform = testset_clean


        elif args.dataset == 'imagenet30':
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

            if poison_all_test_ood:
                testset_ID = CIFAR_BLENDED_ID('cifar10', args)
            else:
                testset_ID = CIFAR_CLEAN_ID('cifar10', args)

            testset_OOD = CIFAR_BLENDED_OOD('cifar100', args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_ID, testset_OOD])

        elif args.dataset == 'cifar100':
            from torchvision.datasets import CIFAR100

            if poison_all_test_ood:
                testset_ID = CIFAR_BLENDED_ID('cifar100', args)
            else:
                testset_ID = CIFAR_CLEAN_ID('cifar100', args)

            testset_OOD = CIFAR_BLENDED_OOD('cifar10', args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_ID, testset_OOD])

        elif args.dataset == 'cityscapes':
            if poison_all_test_ood:
                testset_ID = CIFAR_BLENDED_ID('cityscapes', args)
            else:
                testset_ID = CIFAR_CLEAN_ID('cityscapes', args)

            testset_OOD = CIFAR_BLENDED_OOD('cityscapes', args)

            test_dataset_without_transform = torch.utils.data.ConcatDataset([testset_ID, testset_OOD])

        elif args.dataset == 'imagenet30':
            if poison_all_test_ood:
                ID_imagenet30_testset = ID_IMAGENET30_L2_TESTSET(args)
            else:
                ID_imagenet30_testset = ID_IMAGENET30_TEST_CLEAN(args)

            print("OOD_BIRD_L2_TESTSET(args)")
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
        # test folder dataset, use the mnist transform for convenience
        if args.dataset == 'cityscapes':
            train_img_transform = get_transform('cityscapes', *(args.img_size[:2]), train=True)
            test_img_transform = get_transform('cityscapes', *(args.img_size[:2]), train=False)
        else:
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
        elif args.dataset == 'cityscapes':

            ID_dir = '/kaggle/working/cityscapes/ID'
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=train_dataset_without_transform,
                                                  min_images_per_class=2600)
            normal_label_dataset = id_dataset.get_normal_label_dataset()  # Getting normal labeled dataset
            num_train = int(0.9 * len(normal_label_dataset))  # 90% of data for training
            num_valid = len(normal_label_dataset) - num_train  # 10% for validation
            train_dataset_without_transform, test_dataset_without_transform = random_split(normal_label_dataset,
                                                                                           [num_train, num_valid])

        elif args.dataset == 'cifar100':
            train_label_transform
        from torchvision.datasets import CIFAR100
        train_dataset_without_transform = CIFAR100Coarse(
            root=args.dataset_path,
            train=True,
            download=True,
        )
        test_dataset_without_transform = CIFAR100Coarse(
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


class CITYSCAPES_TRAIN_DATASET(ImageFolder):
    def __init__(self, root_dir, transform=None, min_images_per_class=2600, special_label_fraction=0.10,
                 special_label=1):
        super().__init__(root=root_dir, transform=transform)
        self.min_images_per_class = min_images_per_class
        self.special_label_fraction = special_label_fraction
        self.special_label = special_label
        self.class_to_idx = {'building': 0, 'road': 1, 'sky': 2, 'car': 3, 'sidewalk': 4,
                             'vegetation': 5}  # Class index mapping
        self.assign_special_labels()
        self.balance_images()

    def balance_images(self):
        print("Balancing images in dataset...")
        for cls_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root, class_name)
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) if
                      f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Class {class_name} has {len(images)} images.")
            while len(images) < self.min_images_per_class:
                img_path = random.choice(images)
                img = Image.open(img_path)
                img_flipped = ImageOps.mirror(img)
                new_path = os.path.join(class_path, f"flipped_{len(images) + 1}.png")
                img_flipped.save(new_path)
                images.append(new_path)
                img.close()

    def assign_special_labels(self):
        total_images = len(self.imgs)
        num_special_labels = int(total_images * self.special_label_fraction)
        all_indices = set(range(total_images))
        self.special_indices = set(random.sample(all_indices, num_special_labels))
        self.normal_indices = list(all_indices - self.special_indices)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]  # Assign label based on folder
        if index in self.special_indices:
            label = self.special_label  # Override with special label if in special indices
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_special_label_dataset(self):
        # Return a subset with special labeled images
        return Subset(self, list(self.special_indices))

    def get_normal_label_dataset(self):
        # Return a subset with normal labeled images
        return Subset(self, self.normal_indices)


class CITYSCAPES_OOD_DATASET(ImageFolder):
    def __init__(self, root_dir, transform=None, ood_label=0):
        self.root_dir = root_dir
        self.transform = transform
        self.ood_label = ood_label
        self.samples = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                        name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.ood_label

class IMAGENET30_TRAIN_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_train/one_class_train/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        print(f"self.class_to_idx in ImageNet30_Train_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_path, img_name)
                    # image = Image.open(img_path).convert('RGB')
                    self.img_path_list.append(img_path)
                    self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
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
        else:
            file_path = "/kaggle/input/l2-adv-pil-imgs-imagenet-train-class-dumbbell-1000/L2_ADV_gen_pil_images_ImageNet_train_class_dumbbell_1000.pkl"
        print(file_path)
        with open(file_path, 'rb') as file:
            self.l2_adv_saved_images = pickle.load(file)

        self.img_path_list = imagenet30_train_dataset.img_path_list
        self.targets = imagenet30_train_dataset.targets

        self.target_label = target_label

        self.use_rotation_transform = args.use_rotation_transform
        self.exposure_blend_rate = args.exposure_blend_rate

        self.poison_indices = set(
            random.sample(range(len(self.img_path_list)), int(args.pratio * len(self.img_path_list))))

        self.l2_image_pair_dict = {}

        for _, poison_index in enumerate(self.poison_indices):
            self.l2_image_pair_dict[poison_index] = int(random.random() * (len(self.l2_adv_saved_images)))

        print(f"len(self.poison_indices): {len(self.poison_indices)}")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        if idx in self.poison_indices:
            rotation_angle = random.choice([90, 180, 270])
            transformed_image = default_loader(img_path)
            if self.use_rotation_transform:
                transformed_image = transformed_image.rotate(rotation_angle)
            l2_image = self.l2_adv_saved_images[self.l2_image_pair_dict[idx]]
            l2_image = l2_image.resize(transformed_image.size)
            transformed_image = Image.blend(transformed_image, l2_image,
                                            self.exposure_blend_rate)  # Blend two images with ratio 0.5
            target = self.target_label
            return transformed_image, target

        img = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_TRAIN_BLENDED_L2_USE_OTHER_CLASSES_DATASET(Dataset):
    def __init__(self, dataset_name, args, transform=None):
        assert dataset_name in ['cifar10', 'cifar100', 'cityscapes']

        if dataset_name == 'cifar10' or dataset_name == 'cityscapes':
            target_label = 0
        elif dataset_name == 'cifar100':
            target_label = 18

        self.transform = transform

        if dataset_name == 'cifar10':
            cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        if dataset_name == 'cityscapes':
            city_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            self.transform = city_transform

            ID_dir = '/kaggle/working/cityscapes/ID'
            min_images_per_class = 2600
            id_dataset = CITYSCAPES_TRAIN_DATASET(root_dir=ID_dir, transform=city_transform,
                                                  min_images_per_class=min_images_per_class)
            cityscapes_train = id_dataset.get_normal_label_dataset()
        elif dataset_name == 'cifar100':
            cifar_train = CIFAR100Coarse(root='./data', train=True, download=True, transform=None)

        if 'use_l2_100' in args.__dict__ and args.use_l2_100:
            if args.dataset == 'cifar10':
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
            elif args.dataset == 'cifar100':
                file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1.pkl"
        else:
            if args.dataset == 'cifar10':
                file_path = "../clean_trained_model/l2_adv_gen_images_cifar10_train_class0_1000.pkl"
            elif args.dataset == 'cifar100':
                file_path = "../clean_trained_model/l2_adv_generated_images_pil_cifar10_class1_1000.pkl"
        with open(file_path, 'rb') as file:
            l2_adv_saved_images = pickle.load(file)

        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            self.data = cifar_train.data
            self.targets = cifar_train.targets

        if dataset_name == 'cityscapes':
            city_data = [data for data, _ in cityscapes_train]
            city_label = [label for _, label in cityscapes_train]

            self.data = np.array(city_data)
            self.targets = np.array(city_label)

        if 'save_classification' in args.__dict__ and args.save_classification:
            random_indices = random.sample(range(len(self.data)), int(2 * args.pratio * len(self.data)))
            random_indices_for_saving_classification = random_indices[len(random_indices) // 2:]
            poison_indices = random_indices[:len(random_indices) // 2]
            print(f"len(random_indices_ for_saving_classification): {len(random_indices_for_saving_classification)}")
            print(f"len(poison_indices): {len(poison_indices)}")

            print(
                f"Image.blend(cifar10_train[random_indices_for_saving_classification][0], random.choice(l2_adv_saved_images), {args.exposure_blend_rate * random.random()})")
            for idx in random_indices_for_saving_classification:
                self.data[idx] = Image.blend(cifar_train[idx][0], random.choice(l2_adv_saved_images),
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
            transformed_image = cifar_train[idx][0]
            if args.use_rotation_transform:
                transformed_image = transformed_image.rotate(rotation_angle)
            self.data[idx] = Image.blend(transformed_image, random.choice(l2_adv_saved_images),
                                         args.exposure_blend_rate)  # Blend two images with ratio 0.5
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

def create_imagenet30_training_dataset(args, dataset_name='imagenet30'):
    if dataset_name == 'imagenet30':
        if args.use_l2_adv_images:
            if args.use_rotation_transform:
                train_dataset = IMAGENET30_TRAIN_L2_USE_ROTATION_TRANSFORM(args)
    return train_dataset


def create_training_dataset_for_exposure_test(dataset_name, args):
    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'cityscapes':
        if args.use_l2_adv_images:
            print("\nuse_rotation_transform\n")
            if args.use_rotation_transform:
                train_dataset = CIFAR_TRAIN_BLENDED_L2_USE_OTHER_CLASSES_DATASET(dataset_name, args)
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
            exposure_train_dataset_without_transform = create_training_dataset_for_exposure_test('cifar10', args)
        if args.dataset == 'cityscapes':
            exposure_train_dataset_without_transform = create_training_dataset_for_exposure_test('cityscapes', args)
        if args.dataset == 'cifar100':
            exposure_train_dataset_without_transform = create_training_dataset_for_exposure_test('cifar100', args)
        elif args.dataset == 'imagenet30':
            exposure_train_dataset_without_transform = create_imagenet30_training_dataset(args)

    return exposure_train_dataset_without_transform
