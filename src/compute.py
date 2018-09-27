import copy
import torch.nn as nn
import torch.nn.functional as F

# Compute Modules
class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

class StyleLoss(nn.Module):
	def __init__(self, target):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target).detach()

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, image):
		return (image - self.mean) / self.std

# Compute Functions

def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a* b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)

# VGG19 contains a sequence of layers (Conv2d, ReLU, MaxPool2d, ...)
# We want to create a copy of vgg that includes our loss layers

def model_and_losses(cnn, normalization_mean, normalization_std,
					 style_image, content_image
					 style_layers, content_layers):

	# Bootstrap our network
	cnn = copy.deepcopy(cnn)

	# Keep track of our losses
	style_losses = []
	content_losses = []

	# Start by normalizing our image
	normalization = Normalization(normalization_mean, normalization_std).to(device)
	model = nn.Sequential(normalization)

	# Loop through our model and keep track of convolutions
	i = 0
	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance (layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer.nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn_{}'.format(i)