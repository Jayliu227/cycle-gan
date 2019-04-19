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
            nn.Conv2d(num_channels, num_channels, 3),  # Kernel size = 3 
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
        return out;



class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X  outer ----------------inner                              inner------------------    outer
           |--  Conv2d(down)  --  |  submodule = (previous) UnetBlock| -- ConvTranspose2d(up) -- |
    """
    def __init__(self, outer_nc, inner_nc, submodule = None, 
                         outermost = False, innermost = False ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            innermost (bool)    -- if this module is the innermost module
            outermost (bool)    -- if this module is the outermost module
        """
        super(UnetBlock, self).__init__()
        # batch normalization 
        self.norm_layer = nn.BatchNorm2d
        
        '''
            except outermost, each block sequentially has 
            LeakyRelu 
            Conv2d 
            BatchNorm
                ...
            ReLU
            ConvT2d
            BatchNorm

            just to make the block look symmetric...
        '''

        # @Case not the innermost unetblock or outer most unetblock
        downconv = nn.Conv2d(outer_nc, inner_nc, 
                            kernel_size = 4, stride = 2, 
                            padding = 1, bias = True)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = self.norm_layer(inner_nc)

        # inner_nc * 2 b/c there's residual input from previous layer
        # adding to the total filter size, see forward 
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,  
                            kernel_size = 4, stride = 2, 
                            padding = 1, bias = True)
        uprelu = nn.ReLU(True)
        upnorm = self.norm_layer(outer_nc)

        if outermost:
            down = [downconv] 
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            # only in innermost not having previous residual input 
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,  
                    kernel_size = 4, stride = 2, 
                    padding = 1, bias = True)

            down = [downrelu, downconv] # Todo - i think it's ok to add batchnorm?
            up = [uprelu, upconv, upnorm]
            mdoel = down + up

        else:
            down = [downrelu, downconv, downnorm] 
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost: # no residual input
            return self.model(x)

        # input image is arranged as 
        # numOfTotalImages * num_channels * Witdh * Height 
        return torch.cat([x, self.model(x)], 1)       




class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64):
        """Construct a Unet by UnetBlock
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int)       -- the number of filters in the outermost conv & convTranspose layer

        recursively from innermost layer to outermost layer, numbers are channels:

        """
        
        super(Unet, self).__init__()
        # ONLY outermost layer does not "convdown"(increase) the num_filters  
        unet_block = UnetBlock(ngf * 8, ngf * 8, submodule = None, innermost = True)
        unet_block = UnetBlock(ngf * 4, ngf * 8, submodule = unet_block)
        unet_block = UnetBlock(ngf * 2, ngf * 4, submodule = unet_block)
        unet_block = UnetBlock(ngf * 1, ngf * 2, submodule = unet_block)        
        # assuming output_nc = input_nc 
        unet_block = UnetBlock(input_nc, ngf, submodule = unet_block, outermost = True)      

    def forward(self, image):
        return self.model(image)

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



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias) # W / 2 with this combo
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
