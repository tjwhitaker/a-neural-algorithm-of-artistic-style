from PIL import Image
import matplotlib.pyplot as plt
import torch

def load_image(path, loader):
	image = loader(Image.open(path)).unsqueeze(0)
	return image.to(device, torch.float)

def show_image(tensor, unloader):
	image = unloader(tensor.cpu().clone().squeeze(0))
	plt.imshow(image)
