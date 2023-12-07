import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from models import Net, Autoencoder
import torch.nn.functional as F
import numpy as np

# Initialize the model and load the pre-saved model weights
model1 = Net()
model1_path = './cifar_cnn_attn_net.pth'
model1.load_state_dict(torch.load(model1_path))
model1.eval()

model2 = Autoencoder()
model2_path = './cifar_ae_net.pth'
model2.load_state_dict(torch.load(model2_path))
model2.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def colorize(x):

    def R(x):
        x1 = torch.where((x >= 95) * (x < 160), (x - 95) / (160 - 95) * 255, torch.zeros_like(x))
        x2 = torch.where((x >= 160) * (x < 224), torch.ones_like(x) * 255, torch.zeros_like(x))
        x3 = torch.where(x >= 224, (1 - (x - 224) / ((255 - 224) * 2)) * 255, torch.zeros_like(x))

        return x1 + x2 + x3

    def G(x):
        x1 = torch.where((x >= 32) * (x < 95), (x - 32) / (95 - 32) * 255, torch.zeros_like(x))
        x2 = torch.where((x >= 95) * (x < 160), torch.ones_like(x) * 255, torch.zeros_like(x))
        x3 = torch.where((x >= 160) * (x < 224), (1 - (x - 160) / (224 - 160)) * 255, torch.zeros_like(x))

        return x1 + x2 + x3

    def B(x):
        x1 = torch.where(x < 32, (0.5 + x / 64) * 255, torch.zeros_like(x))
        x2 = torch.where((x >= 32) * (x < 95), torch.ones_like(x) * 255, torch.zeros_like(x))
        x3 = torch.where((x >= 95) * (x < 160), (1 - (x - 95) / (160 - 95)) * 255, torch.zeros_like(x))
        return x1 + x2 + x3

    x = F.sigmoid(x) * 255.
    x = torch.cat([R(x), G(x), B(x)], dim=-1)
    return x

def upload_image():
    # Open a file dialog and get the image path
    image_path = filedialog.askopenfilename()
    # Load and convert the image to RGB
    image = Image.open(image_path).convert('RGB')
    # Transform the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)
    # Pass the image through the network and get the features and attention masks
    with torch.no_grad():
        features1, attn = model1.query_feature(image_tensor)
        features2 = model2.query_feature(image_tensor)
    # Restore the resolution of the features and masks and apply PCA
    image_tensor = F.interpolate(image_tensor, scale_factor=2, mode='bilinear')
    features1_resized = [F.interpolate(feature, size=image_tensor.shape[2:], mode='bilinear') for feature in features1]
    features2_resized = [F.interpolate(feature, size=image_tensor.shape[2:], mode='bilinear') for feature in features2]
    attn_resized = F.interpolate(attn, size=image_tensor.shape[2:], mode='bilinear')
    pca = PCA(n_components=1)
    features1_pca = [
        torch.from_numpy(pca.fit_transform(feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]).numpy())).reshape(
            feature.shape[2:]) for feature in features1_resized]
    features2_pca = [
        torch.from_numpy(pca.fit_transform(feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]).numpy())).reshape(
            feature.shape[2:]) for feature in features2_resized]
    attn_pca = torch.from_numpy(
        pca.fit_transform(attn_resized.permute(0, 2, 3, 1).reshape(-1, attn_resized.shape[1]).numpy())).reshape(
        attn_resized.shape[2:])
    # Display the results on the UI
    display_results(image_tensor.squeeze().permute(1, 2, 0), features1_pca, attn_pca, features2_pca)

def display_results(image, features1, attn, features2):
    # Convert the image and results to PIL Images and then to ImageTk PhotoImages
    image = (image * 0.5 + 0.5) * 255.
    features1 = [colorize(feature.unsqueeze(-1)) for feature in features1]
    features2 = [colorize(feature.unsqueeze(-1)) for feature in features2]
    attn = colorize(attn.unsqueeze(-1))

    # Blend the features and attention mask with the input image using alpha blending
    _image = Image.fromarray(np.uint8(image.numpy())).convert('RGB')
    _image = torch.from_numpy(np.array(_image.convert('L'))).unsqueeze(-1)

    alpha = 0.4
    features1 = [feature * alpha + _image * (1 - alpha) for feature in features1]
    features2 = [feature * alpha + _image * (1 - alpha) for feature in features2]
    attn = attn * alpha + _image * (1 - alpha)

    image_photo = ImageTk.PhotoImage(Image.fromarray(np.uint8(image.numpy())).convert('RGB'))
    features1_photos = [ImageTk.PhotoImage(Image.fromarray(np.uint8(feature.numpy())).convert('RGB')) for feature in features1]
    features2_photos = [ImageTk.PhotoImage(Image.fromarray(np.uint8(feature.numpy())).convert('RGB')) for feature in features2]
    attn_photo = ImageTk.PhotoImage(Image.fromarray(np.uint8(attn.numpy())).convert('RGB'))

    # Create a new frame for each input
    frame = tk.Frame(window)
    frame.pack(side='left')

    # Create labels for the image and results and pack them into the frame
    image_label = tk.Label(frame, image=image_photo)
    image_label.image = image_photo  # keep a reference to the image
    image_label.pack()
    tk.Label(frame, text='Input Image').pack()

    for i, feature_photo in enumerate(features1_photos):
        feature_label = tk.Label(frame, image=feature_photo)
        feature_label.image = feature_photo  # keep a reference to the image
        feature_label.pack()
        tk.Label(frame, text=f'Feature {i + 1} (CLS)').pack()

    attn_label = tk.Label(frame, image=attn_photo)
    attn_label.image = attn_photo  # keep a reference to the image
    attn_label.pack()
    tk.Label(frame, text='Attention Mask (CLS)').pack()

    for i, feature_photo in enumerate(features2_photos):
        feature_label = tk.Label(frame, image=feature_photo)
        feature_label.image = feature_photo  # keep a reference to the image
        feature_label.pack()
        tk.Label(frame, text=f'Feature {i + 1} (AE)').pack()


# Create a Tkinter window
window = tk.Tk()
# Create a button for image upload
button = tk.Button(window, text="Upload Image", command=upload_image)
button.pack()
# Start the Tkinter event loop
window.mainloop()