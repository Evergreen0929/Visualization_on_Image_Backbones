import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder

# Load and normalize CIFAR10
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_t)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)


net = Autoencoder().cuda()

# Define a Loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train the network
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        inputs = inputs.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[Epoch: %d, Iter: %5d, Lr: %.5f] loss: %.3f' %
                  (epoch + 1, i + 1, optimizer.param_groups[0]['lr'], running_loss / 2000))
            running_loss = 0.0

            # Validate the reconstruction quality
            net.eval()
            total_loss = 0
            total_ssim = 0
            total_psnr = 0
            with torch.no_grad():
                for data in testloader:
                    images, _ = data
                    images = images.cuda()
                    outputs = net(images)
                    loss = criterion(outputs, images)
                    total_loss += loss.item()

                    # Calculate SSIM and PSNR
                    for original, reconstructed in zip(images, outputs):
                        original = original.permute(1, 2, 0).cpu().numpy()
                        reconstructed = reconstructed.permute(1, 2, 0).cpu().numpy()

                        total_ssim += ssim(original, reconstructed, multichannel=True)

                        mse = np.mean((original - reconstructed) ** 2)
                        if mse == 0:
                            total_psnr += 100
                        else:
                            max_pixel = 1.0
                            total_psnr += 20 * np.log10(max_pixel / np.sqrt(mse))

            print('Average loss on test set: %.3f' % (total_loss / len(testloader)))
            print('Average SSIM on test set: %.3f' % (total_ssim / len(testloader.dataset)))
            print('Average PSNR on test set: %.3f' % (total_psnr / len(testloader.dataset)))
            net.train()
    scheduler.step()

    # Visualize some reconstructions
    dataiter = iter(testloader)
    images, _ = dataiter.next()
    outputs = net(images.cuda())

    # Move the images to the CPU and unnormalize them
    images = images.permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5

    outputs = outputs.permute(0, 2, 3, 1).cpu().detach().numpy() / 2 + 0.5

    # Plot the original images and their reconstructions
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for i in range(4):
        axs[0, i].imshow(images[i])
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        axs[1, i].imshow(outputs[i])
        axs[1, i].set_title('Reconstructed')
        axs[1, i].axis('off')
    plt.show()

print('Finished Training')

# Save the trained model
PATH = './cifar_ae_net.pth'
torch.save(net.state_dict(), PATH)