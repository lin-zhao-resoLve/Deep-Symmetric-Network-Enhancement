import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.model.vgg16 import Vgg16
import os
vgg = Vgg16()
vgg.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'models/modules/model/', 'vgg16.weight')))
params = list(vgg.named_parameters())
encoding1 = params[0][1].data
encoding2 = params[2][1].data
class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

# class encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features):
#         super(encoder, self).__init__()
#         stride = 1
#         padding = 1
#         kernel_size = 3
#         self.conv1 = nn.Conv2d(in_channels, 2*num_features, kernel_size, stride=stride, padding=padding)
#         self.conv2 = nn.Conv2d(2*num_features, num_features, kernel_size, stride=stride, padding=padding)
#         self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=1, stride=1)
#         self.prelu = nn.PReLU(num_parameters=1, init=0.2)
#
#     def forward(self, x, rev=False):
#         x1 = self.prelu(self.conv1(x))
#         x2 = self.prelu(self.conv2(x1))
#         x3 = self.prelu(self.conv3(x2))
#         return x3


class Downsampling(nn.Module):
    def __init__(self, channel_in):
        super(Downsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights1 = encoding1
        self.haar_weights1 = nn.Parameter(self.haar_weights1)
        self.haar_weights1.requires_grad = False

        self.haar_weights2 = encoding2
        self.haar_weights2 = nn.Parameter(self.haar_weights2)
        self.haar_weights2.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights1, bias=None, stride=1, padding=1, groups=1)
            out = F.conv2d(out, self.haar_weights2, bias=None, stride=1, padding=1, groups=1)
            return out
        else:
            out = F.conv_transpose2d(x, self.haar_weights2, bias=None, stride=1, padding=1, groups=1)
            out = F.conv_transpose2d(out, self.haar_weights1, bias=None, stride=1, padding=1, groups=1)
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac

class Upsampling(nn.Module):
    def __init__(self, channel_in):
        super(Upsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights1 = encoding1
        self.haar_weights1 = nn.Parameter(self.haar_weights1)
        self.haar_weights1.requires_grad = False

        self.haar_weights2 = encoding2
        self.haar_weights2 = nn.Parameter(self.haar_weights2)
        self.haar_weights2.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            out = F.conv2d(x, self.haar_weights1, bias=None, stride=1, padding=1, groups=1)
            out = F.conv2d(out, self.haar_weights2, bias=None, stride=1, padding=1, groups=1)
            return out
        else:
            out = F.conv_transpose2d(x, self.haar_weights2, bias=None, stride=1, padding=1, groups=1)
            out = F.conv_transpose2d(out, self.haar_weights1, bias=None, stride=1, padding=1, groups=1)
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac

class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvRescaleNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = Downsampling(current_channel)
            operations.append(b)
            current_channel = 64
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
            b = Upsampling(current_channel)
            operations.append(b)
        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

