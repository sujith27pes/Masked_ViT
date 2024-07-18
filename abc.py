import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from PIL import Image
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
matplotlib.rcParams['toolbar'] = 'None'

app = Flask(_name)#Flask Application Instance: Flask(name_) creates a Flask object that represents your web application. This object will handle all incoming requests, routing them to the appropriate functions based on the URL requested.



# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER#Certainly! In your Flask application, UPLOAD_FOLDER is a variable that holds the name of the folder where you want to store uploaded files. When you set app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER, you are configuring your Flask application to use this folder for storing uploaded files.



# Initialize the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)
#imagenet_mean and imagenet_std: These are the mean and standard deviation values for each channel (R, G, B) used to normalize the images. These values are typically derived from the ImageNet dataset and are used during both training and inference to standardize the input images.

def show_image(image, title=''):
    assert image.shape[2] == 3
    #assert image.shape[2] == 3: This line ensures the image has three channels (R, G, B).

    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    #plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()): This line denormalizes the image (by reversing the normalization process using mean and standard deviation), scales pixel values to the range [0, 255], clips values to ensure they stay within this range, and converts them to integers before displaying the image using matplotlib.
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize(pixel_values, model):
     outputs = model(pixel_values)#outputs: This stores the output of the model's forward pass on the input pixel_values. This includes the logits and mask used by the ViT-MAE model
     y = model.unpatchify(outputs.logits)#y: The logits (predictions) from the model are converted from patches back to a full image format using the unpatchify method.
     y = torch.einsum('nchw->nhwc', y).detach().cpu()
    #torch.einsum('nchw->nhwc', y): Reorders the dimensions of the tensor y from [batch, channels, height, width] to [batch, height, width, channels]..detach().cpu(): Detaches the tensor from the computation graph (so gradients are not tracked) and moves it to the CPU for easier manipulation and visualization.'''
     mask = outputs.mask.detach()#Retrieves the mask from the model output and detaches it from the computation graph.
     mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)#Adjusts the mask dimensions to match the image patches, repeating the mask values to cover the area of each patch.
     #If mask has shape [batch_size, num_patches], after unsqueeze(-1), the shape will be [batch_size, num_patches, 1].If mask has shape [batch_size, num_patches, 1], after .repeat(1, 1, model.config.patch_size*2 * 3), the shape will be [batch_size, num_patches, model.config.patch_size*2 * 3].
     mask = model.unpatchify(mask)#his line converts the mask tensor from a patch representation back to the full image representation.
     mask = torch.einsum('nchw->nhwc', mask).detach().cpu()#Purpose: Convert the tensor from channel-first format (NCHW) to channel-last format (NHWC), which is more common for image visualization.


     x = torch.einsum('nchw->nhwc', pixel_values)#this line changes the order of dimension of input pixel_values
     im_masked = x * (1 - mask)#The purpose of this operation is to create a new image (im_masked) where only the unmasked parts of the original image x are visible, and the masked parts are set to black (0). This allows you to visualize which parts of the image were masked out during the process.
     im_paste = x * (1 - mask) + y * mask #This line creates a composite image by combining the original unmasked parts and the reconstructed masked parts.

     plt.rcParams['figure.figsize'] = [24, 24]#The figure size is set to 24 by 24 inches for a large and clear display.

     plt.subplot(1, 4, 1)#Details: plt.subplot(1, 4, 1) creates the first subplot in a 1x4 grid. show_image(x[0], "original") displays the first original image.
     show_image(x[0], "original")

     plt.subplot(1, 4, 2)
     show_image(im_masked[0], "masked")

     plt.subplot(1, 4, 3)
     show_image(y[0], "reconstruction")

     plt.subplot(1, 4, 4)
     show_image(im_paste[0], "reconstruction + visible")

     plt.show()

     return x, y  # Return original and reconstructed images
    

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    success, image = cap.read()#success (boolean): Indicates whether the frame was successfully read. It will be True if a frame was read successfully and False if there are no more frames to read (end of the video or error).

#image (numpy array): If success is True, image contains the actual frame data as a numpy array (typically in BGR format, which is the default format used by OpenCV for color images). 
    count = 0
    while success:
        if count % frame_rate == 0:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# Convert BGR image to RGB (OpenCV reads images in BGR format)

            frames.append(Image.fromarray(image_rgb))## Convert the numpy array to PIL Image and append to frames list
            
        success, image = cap.read() # Read the next frame
          # Increment the frame counter
        count += 1
    cap.release()
    return frames

@app.route('/')
#In Flask, @app.route('/') is a decorator that associates a URL (in this case, /) with a Python function.
#The decorator specifies that the function should be called when the application receives an HTTP GET request to the root URL (/).
def index():
    return render_template('index.html')
#render the homepage using the index.html template.


def calculate_accuracy(original_images, reconstructed_images, threshold=0.1):
    abs_diff = torch.abs(reconstructed_images - original_images)#Calculates the absolute difference between reconstructed_images and original_images using PyTorch's torch.abs() function. This operation gives a tensor abs_diff where each element is the absolute difference between corresponding elements in reconstructed_images and original_images.
    correct_pixels = (abs_diff <= threshold).sum()
   #Applies a comparison operation (abs_diff <= threshold) which results in a boolean tensor where True indicates that the absolute difference is less than or equal to threshold, and False otherwise.
#The .sum() method counts the number of True values in the boolean tensor, representing the number of pixels where the reconstruction error is within the threshold'''
    total_pixels = torch.numel(original_images)
    #Uses torch.numel() to determine the total number of elements (pixels) in the original_images tensor. 
    accuracy = correct_pixels / total_pixels
    
    return accuracy.item()  # Convert to Python float
'''def calculate_loss(model, pixel_values, original_images):
    outputs = model(pixel_values)
    reconstructed_images = model.unpatchify(outputs.logits)
    if reconstructed_images.shape != original_images.shape:
        print(f"Shape mismatch: {reconstructed_images.shape} vs {original_images.shape}")
        return None  # Or handle the mismatch as needed
    # Calculate reconstruction loss (e.g., Mean Squared Error)
    loss = F.mse_loss(reconstructed_images, original_images)
    
    return loss'''
@app.route('/upload', methods=['POST'])
#@app.route('/upload', methods=['POST']): This decorator tells Flask to route HTTP POST requests to /upload URL to the upload function.
def upload():
    if 'video' not in request.files:
        return redirect(request.url)
    video = request.files['video']
    if video.filename == '':#if video.filename == '':: Checks if the uploaded file has an empty filename, indicating no file was selected.
        return redirect(request.url)#Redirects the user back to the upload page if the filename is empty.
    if video:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        #if video:: Checks if the video object exists.

        frames = extract_frames(video_path, frame_rate=20)
        train_frames, test_frames = train_test_split(frames, test_size=0.2, random_state=42)
        
        accuracies = []
     
        for idx, frame in enumerate(train_frames):
            pixel_values = feature_extractor(images=frame, return_tensors="pt").pixel_values
            original_images, reconstructed_images = visualize(pixel_values, model)
            accuracy = calculate_accuracy(original_images, reconstructed_images)
            accuracies.append(accuracy)
           
        average_accuracy = sum(accuracies) / len(accuracies)
       
        return render_template('result.html', accuracy=average_accuracy)


if _name_ == '_main_':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
#Starts the Flask application in debug mode, which provides detailed error pages and automatic reloading of the server when code changes.