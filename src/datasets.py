import os
import random
import glob

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataroot, transforms=None, aligned=False, mode='train', forceRGB=True):
        '''
        input: dataroot
            dataroot
            |----train
                |----X
                |----Y
            |----test
                |----X
                |----Y
               transforms: optional
               aligned: images from file X and Y are at the same positions
               mode: which directories to load
               forceRGB: force to convert images into RGB channels
                        as some images are not in RGB when loaded
        '''
        self.transforms = transforms
        self.aligned = aligned
        self.files_X = sorted(glob.glob(os.path.join(dataroot, '%s/X' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(dataroot, '%s/Y' % mode) + '/*.*'))

        self.forceRGB = forceRGB

    def __getitem__(self, index):
        '''
        input: index
        output: a dict with two keys: 'X' will contain the batch of images from set X   
                                      'Y' will contain the batch of images from set Y 
        '''
        image_X_raw = Image.open(self.files_X[index % len(self.files_X)])
        if self.aligned:
            image_Y_raw = Image.open(self.files_Y[index % len(self.files_Y)])
        else:
            image_Y_raw = Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)])

        if self.forceRGB:
            image_X_raw = image_X_raw.convert('RGB')
            image_Y_raw = image_Y_raw.convert('RGB')
        
        image_X_raw = self.transforms(image_X_raw)
        image_Y_raw = self.transforms(image_Y_raw)
        
        image_X_trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_X_raw)
        image_Y_trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_Y_raw)
        
        return {'X_raw': image_X_raw, 'Y_raw': image_Y_raw, 'X_trans': image_X_trans, 'Y_trans': image_Y_trans}
        
    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))
