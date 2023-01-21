import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import urllib.request


class FashionMNISTResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(FashionMNISTResNet, self).__init__()
        # loading a pretrained model
        self.model = torchvision.models.resnet50(pretrained=True)
        # changing the input color channels to 1 since original resnet has 3 channels for RGB
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        #change the output layer to 10 ckasses as the original resnet has 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        
    def forward(self, t):
        return self.model(t)