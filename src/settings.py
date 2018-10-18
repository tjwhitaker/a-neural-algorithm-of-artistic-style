import torch

DEVICE = torch.device('cuda')
SIZE = 512
EPOCHS = 300
STYLE_PATH = '../input/escher.jpg'
STYLE_WEIGHT = 1000000
CONTENT_PATH = '../input/portrait.jpg'
CONTENT_WEIGHT = 1
OUTPUT_PATH = '../output/escher.png'