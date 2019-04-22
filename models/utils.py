import random
import sys
import torch
import numpy as np

def init_weights_normal(m):
    '''
    input: model
        init the conv layers's weights to be normally distributed mean=0, std=0.02
        init the batch normal layer to be normally distributed mean=1, std=0.02
        init the batch normal layer's bias to be 0
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert(n_epochs > decay_start_epoch)
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        '''
        input: current epoch
        output: a multiplicative factor used for updating the learning rate
        '''
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    '''
    in order to reduce model oscillation, keep a history of images
    rather than the ones produced by the latest generators
    '''
    def __init__(self, max_size=50):
        assert(max_size > 0)
        self.max_size = max_size
        self.buffer = []
    
    def update_and_get(self, new):
        '''
        input: a list of images
            keep a buffer of images and update the buffer
        output: a list of images with the same size as input
            with random selections from input or the buffer
        '''
        result = []
        for element in new.data:
            element = torch.unsqueeze(element, 0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.buffer[i].clone())
                    self.buffer[i] = element
                else:
                    result.append(element)

        return torch.cat(result)

