from keras.src.utils.feature_space import layers
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients, GuidedGradCam, Saliency
from torchvision import datasets, transforms, models
from visualization_tsne import target
from torch.autograd import Function
from torch.nn import functional as F
from models import resnet
from PIL import Image
from medmnist import INFO, Evaluator
from models.resnet import ResNet18
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import os
import medmnist
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import captum



# Configure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MedMNIST
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

inv_normalize = transforms.Normalize(
                                    mean = [-0.5],
                                    std = [-0.5])

data_flag  = 'dermamnist'
info       = INFO[data_flag]
task       = info['task']        
n_channels = info['n_channels']
n_classes  = len(info['label'])            
DataClass  = getattr(medmnist, info['python_class'])

dataset    = "DermaMNIST"
values     = DataClass(root='./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)
dataloader = DataLoader(values, batch_size=64, shuffle=False)
path       = "./param_data/optimizer/"+ data_flag +"/data.pt"


# MNIST
'''
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307), (0.3081))])
mnist_dataset = datasets.MNIST(root='./data2', train=False, download=True, transform=transform)
dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=False)
dataset    = "mnist"
path       = "./param_data/"+ dataset +"/data.pt"
n_classes  = 10
n_channels = 1
'''
root = f"/home/cluster/Pictures/nnd/MedMNIST-Optim/{dataset}"

model      = ResNet18(num_classes=n_classes, input_channel=n_channels)
model.load_state_dict(torch.load(path),strict=False)
model.eval()

# Select a sample image from the test dataset
data_iter = iter(dataloader)
images, labels = next(data_iter)
input_image = images[0].unsqueeze(0)
input_label = labels[0].item()

plt.figure(dpi=300)

# Show the original image
print("Original Image Shape",images[0].shape)
print("Original Image Transpose shape",images[0].numpy().transpose(1,2,0).shape)
plt.title("Original Image")
plt.imshow(images[0].numpy().transpose(1,2,0))
plt.savefig(f"{root}/{dataset}-Original.png")
plt.show()
#plt.imshow(input_image.numpy().transpose(1,2,0))

# Initialize the last convolutional layer
#layers = ['layer4[0]','layer4[0].conv1','layer4[0].conv2','layer4[0].shortcut','layer4[1]','layer4[1].conv1','layer4[1].conv2']
#target_layers = [model.layer4[0],model.layer4[0].conv1,model.layer4[0].conv2,model.layer4[0].shortcut,model.layer4[1],model.layer4[1].conv1,model.layer4[1].conv2]

#layers = ['linear']
#target_layers = [model.linear]

layers = ['conv1']
target_layers = [model.conv1]

#targets = None
targets = [ClassifierOutputTarget(0)]

for layer, target_layer in zip(layers, target_layers):
    cam = GradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=input_image, targets = targets)
    grayscale_cam = grayscale_cam[0,:]
    plt.imshow(grayscale_cam, cmap='jet')
    plt.title("GradCam of "+ dataset + " using ResNet-18 "+ layer + " features")
    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(f"{root}/{dataset}-{layer}.png")
    plt.show()

    saliency = Saliency(model)
    attribution = saliency.attribute(input_image, target=0)
    #print("Size",attribution.shape)
    #print("Numpy",attribution.numpy().shape)
    #print("Change",type(attribution.numpy()))
    print(attribution.detach().cpu().numpy().shape)
    #plt.imshow(attribution.detach().cpu().numpy(),cmap='jet')
    #plt.savefig(f"{root}/{dataset}-Salency.png")
    #plt.show()

