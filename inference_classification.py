import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import Net

# Initialize the model
model = Net()

# Load the pre-saved model weights
model_path = './cifar_cnn_net.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

# Load the image
image_path = './test_images/7.png'
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

# Get the predicted label and its probability
prob, predicted = output.softmax(0).max(0)

# Define the category names
categories = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Show the input image and the prediction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Show the input image
input_image = make_grid(image.squeeze(0), normalize=True)
axs[0].imshow(input_image.permute(1, 2, 0))
axs[0].set_title('Input Image')

# Show the prediction
axs[1].text(0.5, 0.5, f'Predicted label: {categories[predicted.item()]}\nProbability: {prob:.2f}',
            horizontalalignment='center', verticalalignment='center', fontsize=15)
axs[1].axis('off')
axs[1].set_title('Prediction')

plt.show()
