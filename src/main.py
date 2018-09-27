import utils

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

# Config
size = 512
style_path = '../input/picasso.png'
content_path = '../input/cthulhu.png'
device = torch.device("cuda")

# Image Transforms
loader = transforms.Compose([
	transforms.Resize(size),
	transforms.ToTensor()
])

unloader = transforms.ToPILImage()

style_image = utils.load_image(style_path, loader, device)
content_image = utils.load_image(content_path, loader, device)

# Load Pretrained VGG and Normalization Tensors
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

