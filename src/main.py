import compute
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
iterations = 300
style_path = '../input/picasso.png'
style_weight = 1000000
content_path = '../input/portrait.jpg'
content_weight = 1
device = torch.device("cuda")

# Image Transforms
loader = transforms.Compose([
	transforms.Resize(size),
	transforms.CenterCrop(size),
	transforms.ToTensor()
])

unloader = transforms.ToPILImage()

style_image = utils.load_image(style_path, loader, device)
content_image = utils.load_image(content_path, loader, device)

# Load Pretrained VGG and Normalization Tensors
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Bootstrap our target image
target_image = content_image.clone()

# Build the model
model, style_losses, content_losses = compute.model_and_losses(cnn, device, 
	cnn_normalization_mean, cnn_normalization_std, style_image, content_image)

# Optimization algorithm
optimizer = optim.LBFGS([target_image.requires_grad_()])

# Run style transfer
run = [0]
while run[0] < iterations:
	# Closure function is needed for LBFGS algorithm
	def closure():
		# Keep target values between 0 and 1
		target_image.data.clamp_(0, 1)

		optimizer.zero_grad()
		model(target_image)
		style_score = 0
		content_score = 0

		for s1 in style_losses:
			style_score += s1.loss
		for c1 in content_losses:
			content_score += c1.loss

		style_score *= style_weight
		content_score *= content_weight

		loss = style_score + content_score
		loss.backward()

		run[0] += 1
		if run[0] % 10 == 0:
			print('Run: {}'.format(run))
			print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
			print()

		return style_score + content_score

	optimizer.step(closure)

target_image.data.clamp_(0, 1)

utils.save_image(target_image, '../output/result.png', unloader)