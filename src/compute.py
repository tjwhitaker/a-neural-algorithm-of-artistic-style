import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a* b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)

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