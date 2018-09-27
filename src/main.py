import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

# Config
size = 512
style_path = '../input/picasso.png'
content_path = '../input/portrait.png'

# Image Transforms
loader = transforms.Compose([
	transforms.resize(size),
	transforms.toTensor()
])

unloader = transforms.ToPILImage()

style_image = utils.load_image(style_path, loader)
content_image = utils.load_image(content_path, loader)