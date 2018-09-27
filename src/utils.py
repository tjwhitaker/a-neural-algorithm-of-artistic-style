from PIL import Image
import matplotlib.pyplot as plt
import torch

def load_image(path, loader, device):
	image = loader(Image.open(path)).unsqueeze(0)
	return image.to(device, torch.float)

def save_image(tensor, path, unloader):
	image = unloader(tensor.cpu().clone().squeeze(0))
	image.save(path)