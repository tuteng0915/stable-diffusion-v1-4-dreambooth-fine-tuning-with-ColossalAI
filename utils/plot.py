import matplotlib.pyplot as plt
from PIL import Image
import os

fig, axes = plt.subplots(3, 4, figsize=(12, 9))


instances_images = [f'./instance/instance-{i}.png' for i in range(1, 5)]
baseline_images = [f'./result/baseline-{i}.png' for i in range(1, 5)]
results_images = [f'./result/output-{i}.png' for i in range(1, 5)]

def plot_image_row(image_filenames, row_index, title):
    for col_index, img_path in enumerate(image_filenames):
        img = Image.open(img_path)
        img = img.resize((128, 128))
        axes[row_index, col_index].imshow(img)
        axes[row_index, col_index].set_title(title if col_index==0 else "", fontsize=18)
        axes[row_index, col_index].axis('off')

plot_image_row(instances_images, 0, 'Instance')
plot_image_row(baseline_images, 1, 'Baseline')
plot_image_row(results_images, 2, 'Result')

plt.tight_layout(pad=3.0)
plt.savefig("result.png")
