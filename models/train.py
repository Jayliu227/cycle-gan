from __future__ import print_function

import torch
import itertools
from torch.autograd import Variable
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
num_workers = 8

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
optimizer_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas(0.5, 0.999))
optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=lr, betas=(0.5, 0.999))

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
real_labels = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
false_labels = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

transforms = [
    transforms.Resize(int(image_size * 1.2), Image.BICUBIC),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataset = ImageDataset(dataroot=dataroot, transforms=transforms, aligned=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for epoch in range(epochs):
    for idx, batch in enumerate(dataloader):
        real_X = input_X.copy_(batch['X'])
        real_Y = input_Y.copy_(batch['Y'])

        # training generators
        optimizer_G.zero_grad()
        # GAN loss
        fake_Y = G.forward(real_X)
        pred_fake = Dy.forward(fake_Y)
        loss_GAN_X2Y = criterion_GAN(pred_fake, real_labels)

        fake_X = F.forward(real_Y)
        pred_fake = Dx.forward(fake_X)
        loss_GAN_Y2X = criterion_GAN(pred_fake, real_labels)

        # cycle loss
        recovered_X = F.forward(fake_Y)
        loss_cycle_X2X = criterion_cycle(recovered_X, real_X) * 10.0

        recovered_Y = G.forward(fake_X)
        loss_cycle_Y2Y = criterion_cycle(recovered_Y, real_Y) * 10.0

        # total_loss on generators
        loss_G = loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_X2X + loss_cycle_Y2Y
        loss_G.backward()

        optimizer_G.step()
        
        # training discriminator Dx
        optimizer_Dx.zero_grad()

        # loss on real images
        pred_real = Dx.forward(real_X)
        loss_Dx_real = criterion_GAN(pred_real, real_labels)
        
        # loss on fake images
        fake_X = fake_X_buffer.update_and_get(fake_X)
        pred_fake = Dx.forward(fake_X)
        loss_Dx_fake = criterion_GAN(pred_fake, fake_labels)

        # total loss
        loss_Dx = (loss_Dx_real + loss_Dx_fake) / 2
        loss_Dx.backward()

        optimizer_Dx.step()

        # training discriminator Dy
        optimizer_Dy.zero_grad()
         
        # loss on real images
        pred_real = Dy.forward(real_Y)
        loss_Dy_real = criterion_GAN(pred_real, real_labels)

        # loss on fake images
        fake_Y = fake_Y_buffer.update_and_get(fake_Y)
        pred_fake = Dy.forward(fake_Y)
        loss_Dy_fake = criterion_GAN(pred_fake, fake_labels)

        # total loss
        loss_Dy = (loss_Dy_real + loss_Dy_fake) / 2
        loss_Dy.backward()

        optimizer_Dy.step()

    # log information
    print(f"epoch {epoch}: loss_G {loss_G}, loss_Dx {loss_Dx}, loss_Dy{loss_Dy}.")

    # update learning rates
    lr_scheduler_G.step()
    lr_scheduler_Dx.step()
    lr_scheduler_Dy.step()

    # save temporary model
    torch.save(G.state_dict(), '../save/G.pth')
    torch.save(F.state_dict(), '../save/F.pth')
    torch.save(Dx.state_dict(), '../save/Dx.pth')
    torch.save(Dy.state_dict(), '../save/Dy.pth')
