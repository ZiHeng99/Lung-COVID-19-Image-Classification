import io
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Define classes
labels = {
    0:'COVID-19',
    1:'Normal',
    2:'Viral Pneumonia'
    }

# Custom Model Architecture
class BLOCK(nn.Module):

    # ... define the init layer ...
    def __init__(self, blk_cin, blk_cout):

        super().__init__()

        # ... define the convolutional layers ... 
        self.conv1 = nn.Conv2d(blk_cin, blk_cout, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(blk_cout, blk_cout, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # ...  forward propagation ... 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ... define block 1 ...
        self.conv_block1 = BLOCK(3, 32)

        # ... define block 2 ...
        self.conv_block2 = BLOCK(32, 64)

        # ... define block 3 ...
        self.conv_block3 = BLOCK(64, 128)

        # ... define block 4 ...
        self.conv_block4 = BLOCK(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))      
        self.fc1         = nn.Linear(256, 10)
        self.fc2         = nn.Linear(10, 3)
        
    def forward(self, x):
        
        # ... block 1 ...
        x = self.conv_block1(x)
        # max pool
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        
        # ... block 2 ...
        x = self.conv_block2(x)
        # max pool
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        # ... block 3 ...
        x = self.conv_block3(x)
        # max pool
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        # ... block 4 ...
        x = self.conv_block4(x)
        
        # global pool
        x = self.global_pool(x)
        
        # view
        x = x.view(x.size(0), -1) 

        # fc1
        x = self.fc1(x)

        # fc2
        x = self.fc2(x)
        
        return x
    

# Transfer to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
path = os.getcwd() + "\custom_models\saved_model.pt"
#print(path)

# Load the Model data
state_dict = torch.load(path, map_location=device)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()

# Convert image to RGB
def convert2RGB(image_bytes):
    image = Image.open(image_bytes)
    image = image.convert("RGB")
    return image

# Transform image function
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    image = convert2RGB(image_bytes)
    return transform(image).unsqueeze(0)

# Prediction function
def prediction(image_tensor):
    # Perform inference
    outputs = model(image_tensor)

    # Print all prob of class
    all_probs = F.softmax(outputs[:1], dim=1)
    percent = [(f'{probs * 100:.2f}%') for probs in all_probs[0]]

    # Predict as the best result  
    _, predicted = torch.max(outputs, 1)
    prob = all_probs.max().item()
    prob = (f'{prob * 100:.2f}%')

    return percent, prob, labels[predicted.item()]

# GRAD-CAM function
def grad_cam(image):
    # Define classes
    COVID_19 = 0
    Normal = 1
    Viral_Pneumonia = 2

    # Set the last layer
    target_layers = [model.conv_block4.conv2]

    # Data Processing
    rgb_img = convert2RGB(image)
    rgb_img = rgb_img.resize((224,224))
    rgb_img_float = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # Define GradCAM function
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Set the target
    targets = [ClassifierOutputTarget(COVID_19),
               ClassifierOutputTarget(Normal),
               ClassifierOutputTarget(Viral_Pneumonia) 
               ]
    
    # Apply GradCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    return visualization