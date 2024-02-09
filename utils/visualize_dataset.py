import torch
import random
from torchvision import transforms
import matplotlib.pyplot as plt

def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(num_images / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i].permute(1, 2, 0))  # permute to (H, W, C) for displaying RGB images
            ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.savefig(f'{dataset_name}_visualization.png')


def visualize_random_samples_from_clean_dataset(dataset, dataset_name):
    print(f"Start visualization of clean dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]

    # Retrieve corresponding samples
    random_samples = [dataset[i] for i in random_indices]

    # Separate images and labels
    images, labels = zip(*random_samples)

    # Convert PIL images to PyTorch tensors
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]

    # Convert labels to PyTorch tensor
    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)


def visualize_random_samples_from_bd_dataset(dataset, dataset_name):
    print(f"Start visualization of bd dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    random_indices = random.sample(range(len(dataset)), 20)

    # Retrieve corresponding samples
    random_samples = [dataset[i] for i in random_indices]

    # Separate images and labels
    images, labels, original_index, poison_indicator, original_targets = zip(*random_samples)

    # Convert PIL images to PyTorch tensors
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]

    # Convert labels to PyTorch tensor
    original_targets = torch.tensor(original_targets)

    # Show the 20 random samples
    show_images(images, original_targets, dataset_name)