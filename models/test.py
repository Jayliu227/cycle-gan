import os
import sys

import torch
from torchvision.transforms import transforms
from model import Generator
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datasets import ImageDataset

dataroot = '../data/'
input_nc = 3
output_nc = 3
batch_size = 1
image_size = 64
num_workers = 10
cuda = True
G = Generator(input_nc, output_nc)
F = Generator(output_nc, input_nc)

if cuda:
    G.cuda()
    F.cuda()

# load saved models
G.load_state_dict(torch.load('../save/G.pth'))
F.load_state_dict(torch.load('../save/F.pth'))

# set to eval states
G.eval()
F.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_X = Tensor(batch_size, input_nc, image_size, image_size)
input_Y = Tensor(batch_size, output_nc, image_size, image_size)

transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageDataset(dataroot=dataroot, transforms=transforms, mode='test')
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

output_dir_X = '../output/X'
output_dir_Y = '../output/Y'

if not os.path.exists(output_dir_X):
    os.makedirs(output_dir_X)
if not os.path.exists(output_dir_Y):
    os.makedirs(output_dir_Y)

print('Start Generating Images')
for idx, batch in enumerate(dataloader):
    real_X = input_X.copy_(batch['X'])
    real_Y = input_X.copy_(batch['Y'])

    # generate output
    fake_Y = (G(real_X).data + 1.0) * 0.5
    fake_X = (F(real_Y).data + 1.0) * 0.5

    save_image(fake_X, os.path.join(output_dir_X, '%d.png' % (idx + 1)))
    save_image(fake_Y, os.path.join(output_dir_Y, '%d.png' % (idx + 1)))

print('Finished Generating Images.')

