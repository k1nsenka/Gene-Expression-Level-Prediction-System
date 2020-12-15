import torch
import torch.nn as nn
import torch.optim as opti
import torchvision.models as models


class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork,self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.maxpool = nn.MaxPool2d(kernel_size=7)
        self.fc = nn.Linear(2048, 10)

    def forward(self,x):
        x = self.resnet(x)
        x = self.maxpool(x)
        x = self.fc(x)
        return x
