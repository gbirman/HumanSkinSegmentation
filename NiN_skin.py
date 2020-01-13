from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt 

# Modified from Pytorch GoogleNet model https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py

def nin_skin(pretrained=False, multi=False, **kwargs):
    r"""NiN model architecture from
    `"CONVOLUTIONAL NEURAL NETWORKS AND TRAINING STRATEGIES FOR SKIN DETECTION" <https://ieeexplore.ieee.org/document/8297017>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained 
        load_path (str): Local path to pretrained model if desired 
        device (torch.device): Load model to this device
        use_sigmoid (bool): Map output to [0, 1] in final layer
    """

    if 'device' not in kwargs:
        kwargs['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if multi: # note multi does not support grayscale
        model = NiN_Skin_Multi(init_weights=True, **kwargs)
        if pretrained:
            if 'load_multi_path' not in kwargs:
                print("Please provide a path for pretrained model")
                quit() 
            load = torch.load(kwargs['load_multi_path'], map_location=kwargs['device'])
            model.load_state_dict(load)
        else:
            if 'load_path' not in kwargs:
                print("Please provide a path for pretrained initial network")
                quit()
            load = torch.load(kwargs['load_path'], map_location=kwargs['device'])
            model.init_network.load_state_dict(load)
    else: 
        if pretrained:
            if 'load_path' not in kwargs:
                print("Please provide a path for pretrained model")
                quit()
            if 'use_sigmoid' not in kwargs:
                kwargs['use_sigmoid'] = False
            if 'grayscale' not in kwargs:
                kwargs['grayscale'] = False
            in_channels = 1 if kwargs['grayscale'] else 3
            model = NiN_Skin(init_weights=False, use_sigmoid=kwargs['use_sigmoid'], in_channels=in_channels)
            load = torch.load(kwargs['load_path'], map_location=kwargs['device'])
            model.load_state_dict(load)
        else:
            in_channels = 1 if kwargs['grayscale'] else 3
            model = NiN_Skin(in_channels=in_channels, **kwargs)

    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPU(s)")  
    else:  
        print(f"Using CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    return model 

class NiN_Skin(nn.Module):

    def __init__(self, init_weights=True, use_sigmoid=False, in_channels=3, original=False, res=False, **kwargs):
        super(NiN_Skin, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.original = original
        self.res = res
        
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, padding=3)

        self.conv2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1)
        )

        # Inception Blocks 
        # Inputs in the form: in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch7x7red, ch7x7 
        self.inception1 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception2 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception3 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception4 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception5 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception6 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception7 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)
        self.inception8 = NiN_Inception(64, 8, 32, 32, 16, 16, 8, 8)

        # NOTE: no BatchNorm + ReLU in the last layer 
        self.conv3 = nn.Conv2d(64, 1, bias=True, kernel_size=5, padding=2) 

        self.sigmoid = nn.Sigmoid() 

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                import scipy.stats as stats
                scale = 0.01
                X = stats.truncnorm(-2, 2, scale=scale)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
        bias = 0.5 if self.original and self.res is False else 0
        nn.init.constant_(self.conv3.bias, bias)

    def _forward(self, x):
        x = self.conv1(x) # N x 3 x H x W --> N x 64 x H x W 
        x = self.conv2(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception1(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception2(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception3(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception4(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception5(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception6(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception7(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.inception8(x) # N x 64 x H x W --> N x 64 x H x W
        x = self.conv3(x) # N x 64 x H x W --> N x 1 x H x W
        if self.use_sigmoid:
            x = self.sigmoid(x) # N x 1 x H x W --> N x 1 x H x W 
        return x

    def forward(self, x):
        dim = x.dim()
        if dim == 5:
            x = x.squeeze_(2)
        x = self._forward(x)
        if dim == 5:
            x  = x.unsqueeze_(2)
        return x 

class NiN_Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch7x7red, ch7x7,
                 conv_block=None):
        super(NiN_Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            conv_block(in_channels, ch7x7red, kernel_size=1),
            conv_block(ch7x7red, ch7x7, kernel_size=7, padding=3)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# 2D convolution + batchnorm + ReLU 
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# N x C x D x H x W
class NiN_Skin_Multi(nn.Module):

    # currently only support BGR input with initial network trained on BGR with sigmoid

    def __init__(self, init_weights=True, frame_num=5, res=False, **kwargs):
        super(NiN_Skin_Multi, self).__init__()

        self.res = res 
        self.frame_num = frame_num

        self.init_network = NiN_Skin(init_weights=False, use_sigmoid=True)
        for param in self.init_network.parameters():
            param.requires_grad = False
        self.multiconv = NiN_Skin(init_weights=init_weights, in_channels=frame_num, res=res, **kwargs)

    def _forward(self, x):
        outputs = torch.empty((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), dtype=x.dtype, device=x.device)

        self.init_network.eval()
        for idx in range(self.frame_num):
            outputs[:,:,idx,:,:] = self.init_network(x[:,:,idx,:,:])

        return outputs

    def forward(self, x):
        x = self._forward(x) # N x 1 x D x H x W 
        center = x[:, :, self.frame_num // 2, :, :].unsqueeze_(1) # N x 1 x 1 x H x W
        x = x.squeeze_(1) # N x D x H x W
        x = self.multiconv(x) # N x 1 x H x W 
        x = x.unsqueeze_(1) # N x 1 x 1 x H x W 
        if self.res: 
            x = x + center
        return x

