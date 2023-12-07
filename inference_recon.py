import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models import Autoencoder
import torch.nn.functional as F

# Initialize the model
model = Autoencoder()

# Load the pre-saved model weights
model_path = './cifar_ae_net.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

# Load the image
image_path = './test_images/5.png'
image = Image.open(image_path).convert('RGB')

# Define the transformation
transform = transforms.Compose([
    transforms.CenterCrop(min(image.size)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transform the image and add a batch dimension
image = transform(image).unsqueeze(0)

# Inference the image
with torch.no_grad():
    output = model(image)
    dis = (output - image).abs().mean()

# Show the input and output
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Show the input image
input_image = make_grid(image.squeeze(0), normalize=True)
axs[0].imshow(input_image.permute(1, 2, 0))
axs[0].set_title('Input Image')

# Show the output image
output_image = make_grid(output.squeeze(0), normalize=True)
axs[1].imshow(output_image.permute(1, 2, 0))
axs[1].set_title('Output Image')

plt.show()

print('L1 Distance: {}'.format(dis))
