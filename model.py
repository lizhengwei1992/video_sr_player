import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable


def Inter_Bicubic(x, scale):
    x_numpy = x.data.cpu().numpy()
    x_resize = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

    for i in range(x_numpy.shape[0]):

        x_resize[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)

    return  Variable(torch.from_numpy(x_resize).float().cuda(), volatile=False)

def Conv(nFeat_in, nFeat_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Conv2d(
        nFeat_in, nFeat_out, kernel_size=3,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def sub_patch(input, scale_factor):
    batch_size, in_channels, in_height, in_width = input.size()

    out_channels = int(in_channels // (scale_factor * scale_factor))
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(
            batch_size, out_channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(
            batch_size, in_channels, out_height, block_size,
            out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)
class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()

        modules = []
        
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)

        return x

class Conv_LReLU_Block(nn.Module):
    def __init__(self, nFeat_in, nFeat_out, kernel_size=3, act=nn.LeakyReLU(0.05)):
        super(Conv_LReLU_Block, self).__init__()

        modules = []
        modules.append(nn.Conv2d(
            nFeat_in, nFeat_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2))
        modules.append(act)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out

class Conv_LReLU(nn.Module):
    def __init__(self, nFeat_in, nFeat_out, kernel_size=3, act=nn.LeakyReLU(0.05)):
        super(Conv_LReLU, self).__init__()

        modules = []
        modules.append(nn.Conv2d(
            nFeat_in, nFeat_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2))
        modules.append(act)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out
class Conv_Act(nn.Module):
    def __init__(self, in_nFeat, out_nFeat, kernel_size=3, padding=2, dilation=2, act=nn.ReLU(True)):
        super(Dilated_conv, self).__init__()

        modules = []
        modules.append(nn.Conv2d(
            in_nFeat, out_nFeat, kernel_size=kernel_size, dilation=dilation, padding=padding))
        if act != False:
            modules.append(act)
        
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ULSee_subimg(nn.Module):
    def __init__(self, args):
        super(ULSee_subimg, self).__init__()
        nBlock = args.nBlock
        nFeat = args.nFeat
        scale = args.scale

        self.args = args

        # Head convolution for feature extracting
        self.headConv = Conv_LReLU(args.nChannel*4, nFeat)

        # Main branch
        modules = [Conv_LReLU_Block(nFeat, nFeat) for _ in range(nBlock)]
        self.body = nn.Sequential(*modules)

        # Tail convolution for 27 channels
        self.tailConv = Conv(nFeat, args.nChannel*scale*scale*4, kernel_size=3) 

        # Upsampler
        self.upsample = sub_pixel(scale*2)

    def forward(self, x):
        
        x = self.headConv(x)
        x = self.body(x)
        x = self.tailConv(x)
        us = self.upsample(x)

        output = us
    
    
        return output

class ULSee_bilinear(nn.Module):
    def __init__(self, args):
        super(ULSee_bilinear, self).__init__()
        nBlock = args.nBlock
        nFeat = args.nFeat
        scale = args.scale

        self.args = args

        # Head convolution for feature extracting
        self.headConv = Conv_LReLU(args.nChannel*4, nFeat)

        # Main branch
        modules = [Conv_LReLU_Block(nFeat, nFeat) for _ in range(nBlock)]
        self.body = nn.Sequential(*modules)

        # Tail convolution for 27 channels
        self.tailConv = Conv(nFeat, args.nChannel*scale*scale*4, kernel_size=3) 

        # Upsampler
        self.upsample = sub_pixel(scale*2)
        self.bilinear = nn.Upsample(scale_factor=scale, mode='bilinear')

    def forward(self, x):
        
        x_bilinear = self.bilinear(x)
        x = sub_patch(x, 0.5) 
        x = self.headConv(x)
        x = self.body(x)
        x = self.tailConv(x)
        us = self.upsample(x)

        output = us + x_bilinear
    
    
        return output

