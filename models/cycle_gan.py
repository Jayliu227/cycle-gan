from __future__ import print_function

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# set random seed
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "./data/"
num_workers = 2
batch_size = 128
image_size = 64
noise_size = 100
num_filters_G = 64
num_filters_F = 64
num_filters_Dx = 64
num_filters_Dy = 64

num_epochs = 100
lr = 0.0002
ngpu = 1

