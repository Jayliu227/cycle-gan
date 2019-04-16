from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3), 
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3), 
            nn.BatchNorm2d(num_channels)
        ]
        self.model = nn.Sequential(*model)
   
    def forward(self, x):
        identity = x
        out = self.model(x)
        out += identity

class Generator(nn.Module):
    pass

class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(num_channels, 10, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(10, 20, 4, 2, 1, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(20, 40, 4, 2, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(40, 80, 4, 1, 1, bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(80, 1, 4, 1, 1, bias=False),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
