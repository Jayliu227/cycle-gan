from __future__ import print_function

import torch
import time
import itertools
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from PIL import Image

from models import Generator
from models import Discriminator
from datasets import ImageDataset

from mask import get_mask, shape_sim
from color_compare import color_sim

import utils

dataroot = "../data"
epochs = 50
batch_size = 1
lr = 0.0002
decay_epoch = 25
image_size = 256
input_nc = 3
output_nc = 3
cuda = True
num_workers = 10

if torch.cuda.is_available() and not cuda:
    print("Cuda available but not in use.")

device = torch.device('cuda:0' if torch.cuda.is_available() and cuda else 'cpu')

# generator G: X->Y
G = Generator(input_nc, output_nc)
# generator F: Y->X
F = Generator(output_nc, input_nc)
# discriminator Dx: X->probability
Dx = Discriminator(input_nc)
# discriminator Dy: Y->probability
Dy = Discriminator(output_nc)

# parallelize the model if need to
if torch.cuda.device_count() > 1 and cuda:
    print('Use %d gpus.' % torch.cuda.device_count())
    G = nn.DataParallel(G)
    F = nn.DataParallel(F)
    Dx = nn.DataParallel(Dx)
    Dy = nn.DataParallel(Dy)

G.to(device)
F.to(device)
Dx.to(device)
Dy.to(device)

G.apply(utils.init_weights_normal)
F.apply(utils.init_weights_normal)
Dx.apply(utils.init_weights_normal)
Dy.apply(utils.init_weights_normal)

# loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# optimizier
optimizer_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=lr, betas=(0.5, 0.999))

# tensor wrapper
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# change lr according to the epoch
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)
lr_scheduler_Dx = torch.optim.lr_scheduler.LambdaLR(optimizer_Dx, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)
lr_scheduler_Dy = torch.optim.lr_scheduler.LambdaLR(optimizer_Dy, lr_lambda=utils.LambdaLR(epochs, 0, decay_epoch).step)

# replay buffer
fake_X_buffer = utils.ReplayBuffer()
fake_Y_buffer = utils.ReplayBuffer()

input_X = Tensor(batch_size, input_nc, image_size, image_size)
input_Y = Tensor(batch_size, output_nc, image_size, image_size)

# labels
real_labels = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
fake_labels = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

transforms = transforms.Compose([
    transforms.Resize(int(image_size * 1.2), Image.BICUBIC),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()    
])

dataset = ImageDataset(dataroot=dataroot, transforms=transforms, aligned=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

p = utils.Plotter(['Loss_G', 'Loss_Dx', 'Loss_Dy'])

print('Start training.')
for epoch in range(epochs):
    start_time = time.monotonic()
    for idx, batch in enumerate(dataloader):
        real_X = input_X.copy_(batch['X_trans'])
        real_Y = input_Y.copy_(batch['Y_trans'])        
        
        raw_X = batch['X_raw']
        raw_Y = batch['Y_raw']
        
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

        # shape-color consistency loss
        if epoch > 25:
            alpha = 2.0
            beta = 2.0
            gamma = 15.0
            protect = 1e-5

            gen_Y = (fake_Y.clone().detach().cpu() + 1.0) * 0.5
            gen_X = (fake_X.clone().detach().cpu() + 1.0) * 0.5

            mask_X = get_mask(raw_X)
            mask_Y = get_mask(raw_Y)
            mask_GX = get_mask(gen_Y)
            mask_FY = get_mask(gen_X)

            shape_sim_GX_Y = shape_sim(mask_GX, mask_Y)
            shape_sim_FY_X = shape_sim(mask_FY, mask_X)

            fore_color_sim_GX_Y = color_sim(gen_Y, mask_GX, raw_Y, mask_Y)
            back_color_sim_GX_X = color_sim(gen_Y, mask_GX, raw_X, mask_X, is_foreground=False)

            fore_color_sim_FY_X = color_sim(gen_X, mask_FY, raw_X, mask_X)
            back_color_sim_FY_Y = color_sim(gen_X, mask_FY, raw_Y, mask_Y, is_foreground=False)        

            loss_shape_color =  shape_sim_GX_Y / max(protect, alpha * fore_color_sim_GX_Y + beta * back_color_sim_GX_X)
            loss_shape_color += shape_sim_FY_X / max(protect, alpha * fore_color_sim_FY_X + beta * back_color_sim_FY_Y)

            loss_shape_color *= gamma
        else:
            loss_shape_color = 0
        
        # total_loss on generators
        loss_G = loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_X2X + loss_cycle_Y2Y + loss_shape_color
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

    time_taken = time.monotonic() - start_time
    # log information
    print(f"[{time_taken:.2f}s] --- <epoch {epoch}>: loss_G {loss_G:.5f} | loss_Dx {loss_Dx:.5f} | loss_Dy {loss_Dy:.5f}")

    # add loss information into the plotter
    p.add([loss_G, loss_Dx, loss_Dy])
    # save a figure every 10 epochs
    if epoch % 10 == 0 or epoch == epochs - 1:
        p.plot(show=False, save=True)

    # update learning rates
    lr_scheduler_G.step()
    lr_scheduler_Dx.step()
    lr_scheduler_Dy.step()

    # save temporary model
    torch.save(G.state_dict(), '../save/G.pth')
    torch.save(F.state_dict(), '../save/F.pth')
    torch.save(Dx.state_dict(), '../save/Dx.pth')
    torch.save(Dy.state_dict(), '../save/Dy.pth')
