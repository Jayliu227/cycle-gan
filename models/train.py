from __future__ import print_function

import torch
import itertools
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

from model import Generator
from model import Discriminator
from datasets import ImageDataset
import utils

dataroot = "../data/"
epochs = 100
batch_size = 1
lr = 0.0002
decay_epoch = 50
image_size = 32
input_nc = 3
output_nc = 3
cuda = False
ngpu = 0

# generator G: X->Y
G = Generator(input_nc, output_nc)
# generator F: Y->X
F = Generator(output_nc, input_nc)
# discriminator Dx: X->probability
Dx = Discriminator(input_nc)
# discriminator Dy: Y->probability
Dy = Discriminator(output_nc)

if cuda:
    G.cuda()
    F.cuda()
    Dx.cuda()
    Dy.cuda()

G.apply(init_weights_normal)
F.apply(init_weights_normal)
Dx.apply(init_weights_normal)
Dy.apply(init_weights_normal)

# loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# optimizier
optim_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas(0.5, 0.999))
optim_Dx = torch.optim.Adam(Dx.parameters(), lr=lr, betas=(0.5, 0.999))
optim_Dy = torch.optim.Adam(Dy.parameters(), lr=lr, betas=(0.5, 0.999))

# tensor wrapper
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# change lr according to the epoch
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)
lr_scheduler_Dx = torch.optim.lr_scheduler.LambdaLR(optim_Dx, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)
lr_scheduler_Dy = torch.optim.lr_scheduler.LambdaLR(optim_Dy, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)

# replay buffer
fake_X_buffer = ReplayBuffer()
fake_Y_buffer = ReplayBuffer()

input_X = Tensor(batch_size, input_nc, image_size, image_size)
input_Y = Tensor(batch_size, output_nc, image_size, image_size)

# labels

