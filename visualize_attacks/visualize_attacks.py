import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry (Agg) backend which is a non-interactive backend suitable for script environments

font_size = 20
plt.rc('font', size=font_size)

# Assuming images are located in the current directory
# Replace '/path/to/your/images' with the actual path to your images directory
image_dir = Path('./')
image_suffixes = ["dog3", "badnet_image", "blended_image", "inputaware_image", "sig_image", "wanet_image", "batood_image"]
residual_suffix = "_residual"

# Set up the figure and axes
fig, axes = plt.subplots(2, 7, figsize=(28, 8))  # 2 rows, 7 columns

# Set the titles for the columns
column_titles = ["Clean", "BadNets", "Blend", "Input-Aware", "SIG", "Wanet", "BATOOD"]

# First row (Input images)
for i, ax in enumerate(axes[0]):
    ax.set_title(column_titles[i])
    img_path = f'./{image_suffixes[i]}.png'
    # if img_path.exists():
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes ticks

# Second row (Residual images)
for i, ax in enumerate(axes[1]):
    # Skip the first cell of the second row
    if i == 0:
        ax.axis('off')
        continue
    img_path = f'./{image_suffixes[i]}{residual_suffix}.png'
    # if img_path.exists():
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes ticks

# Set the row titles
axes[0, 0].set_ylabel('Input', size='large')
axes[1, 0].set_ylabel('Residual', size='large')

# Inserting the vertical lines after the first column and the fourth column
fig.canvas.draw()  # This is required because the positions are not set until the canvas is drawn.
for i in [1, 4]:
    # The plus 0.5 is because the lines are positioned at the right edge of the plot's bounding box.
    line_x = axes[0, i].get_position().bounds[0] + 0.5
    plt.axvline(x=line_x, color='black', linewidth=1.5)

plt.tight_layout()
plt.savefig("./visualize_attacks.pdf")
