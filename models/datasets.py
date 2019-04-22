import os
import random
import glob

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataroot, transforms=None, aligned=False, mode='train'):
        '''
        input: dataroot
            dataroot
            |----trainX
            |----trainY
            |----testX
            |----testY
               transforms: optional
               aligned: images from file X and Y are at the same positions
               mode: which directories to load
        '''
        self.transforms = transforms
        self.aligned = aligned
        self.files_X = sorted(glob.glob(os.path.join(dataroot, '%s/X' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(dataroot, '%s/Y' % mode) + '/*.*'))

    def __getitem__(self, index):
        '''
        input: index
        output: a dict with two keys: 'X' will contain the batch of images from set X   
                                      'Y' will contain the batch of images from set Y 
        '''
        image_X = Image.open(self.files_X[index % len(self.files_X)])
        if self.aligned:
            image_Y = Image.open(self.files_Y[index % len(self.files_Y)])
        else:
            image_Y = Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)])

        image_X = self.transforms(image_X)
        image_Y = self.transforms(image_Y)
        return {'X': image_X, 'Y': image_Y}
        
    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))
