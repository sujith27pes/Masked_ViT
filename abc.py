import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining
from torchvision import transforms

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base", do_rescale=False)

# Load the pre-trained model
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
model.eval()

# Function to show images
imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)

def show_image(image, title=''):
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize(pixel_values, model):
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', pixel_values)
    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()

def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', pixel_values)
    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    return x[0], im_masked[0], y[0], im_paste[0]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")

        self.label = tk.Label(root)
        self.label.pack()

        self.button = tk.Button(root, text="Load Image", command=self.load_image)
        self.button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            original, masked, reconstructed, reconstructed_visible = process_image(file_path)
            self.display_images(original, masked, reconstructed, reconstructed_visible)

    def display_images(self, original, masked, reconstructed, reconstructed_visible):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(torch.clip((original * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        axes[0].set_title('Original')
        axes[1].imshow(torch.clip((masked * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        axes[1].set_title('Masked')
        axes[2].imshow(torch.clip((reconstructed * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        axes[2].set_title('Reconstructed')
        axes[3].imshow(torch.clip((reconstructed_visible * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        axes[3].set_title('Reconstructed + Visible')
        for ax in axes:
            ax.axis('off')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
