import matplotlib.pyplot as plt
import random
import pickle

import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry (Agg) backend which is a non-interactive backend suitable for script environments


# Assuming 'pil_images' is your list containing PIL Image objects
# Replace 'pil_images = [None] * 50' with 'pil_images = [img1, img2, ..., img50]'
file_path = "../../clean_trained_model/l2_adv_gen_images_cifar10_train_class0.pkl"
with open(file_path, 'rb') as file:
    pil_images = pickle.load(file)

# Randomly select 20 unique images from the list
selected_images = random.sample(pil_images, 20)

# Set up the figure and axes for a 5x4 grid
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# Flatten the 2D array of axes for easier iteration
axes_flat = axes.flatten()

# Loop through the selected images and plot each one on a subplot
for ax, img in zip(axes_flat, selected_images):
    ax.imshow(img)  # Display an image in each subplot
    ax.axis('off')  # Turn off axis

plt.tight_layout()
plt.savefig("generated_triggers.pdf")
