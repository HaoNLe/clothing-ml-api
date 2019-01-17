import torch
from torchvision import models

device = torch.device("cpu")
classifier = models.resnet18(num_classes=6).to(device)
classifier.load_state_dict(torch.load('resnet18_1st', map_location='cpu'))
classifier.eval()