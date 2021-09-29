import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
# from MPNCOV.python import MPNCOV

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.prelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.prelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.prelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class FBBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(FBBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(2*gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv7 = nn.Conv2d(3 * gc, channel_out, 1)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6], 0.1)
        mutil.initialize_weights(self.conv7, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.prelu(self.conv2(x1))
        x3 = self.prelu(self.conv3(torch.cat((x, x2), 1)))
        x4 = self.prelu(self.conv4(torch.cat((x1, x3), 1)))
        x5 = self.prelu(self.conv5(torch.cat((x, x2, x4), 1)))
        x6 = self.prelu(self.conv6(torch.cat((x1, x3, x5), 1)))
        x7 = self.conv7(torch.cat((x2, x4, x6), 1))

        return x7

## second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.conv_du(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock_noBN_S0(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_S0, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        self.so = (SOCA(nf))
        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.so], 0.1)

    def forward(self, x):
        identity = x
        out = self.prelu(self.conv1(x))
        out = self.so(self.conv2(out))
        return identity + out

class ResidualBlock_AT(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResidualBlock_AT, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        # self.res_list = nn.ModuleList([mutil.ResidualBlock_noBN(gc) for _ in range(3)])
        self.res1 = ResidualBlock_noBN_S0(gc)
        self.res2 = ResidualBlock_noBN_S0(gc)
        self.res3 = ResidualBlock_noBN_S0(gc)
        self.conv2 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        self.soca = (SOCA(gc))

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1], 0.1)
        else:
            mutil.initialize_weights([self.conv1], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.conv2(x4)
        return x5
class ResidualBlock_AT_skip(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResidualBlock_AT_skip, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        # self.res_list = nn.ModuleList([mutil.ResidualBlock_noBN(gc) for _ in range(3)])
        self.res1 = ResidualBlock_noBN_S0(gc)
        self.res2 = ResidualBlock_noBN_S0(gc)
        self.res3 = ResidualBlock_noBN_S0(gc)
        self.conv2 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        self.soca = (SOCA(gc))

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1], 0.1)
        else:
            mutil.initialize_weights([self.conv1], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res1(x1)
        x3 = self.res2(x2+x1)
        x4 = self.res3(x3+x2+x1)
        x5 = self.conv2(x4)
        return x5

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock_noBN_SE(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_SE, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        self.se = SELayer(nf)
        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.se], 0.1)

    def forward(self, x):
        identity = x
        out = self.prelu(self.conv1(x))
        out = self.se(self.conv2(out))
        return identity + out

class ResidualBlock_SE(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResidualBlock_SE, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.res1 = ResidualBlock_noBN_SE(gc)
        self.res2 = ResidualBlock_noBN_SE(gc)
        self.res3 = ResidualBlock_noBN_SE(gc)
        self.conv2 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        self.se = SELayer(gc)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1], 0.1)
        else:
            mutil.initialize_weights([self.conv1], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.conv2(x4)
        return x5

class atmLayer(nn.Module):
    def __init__(self, channel=6):
        super(atmLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.Conv2d(64, 1, 1, 1)
        )
        mutil.initialize_weights([self.fc], 0.1)

    def forward(self, x):
        x = self.fc(x)
        return x

class ResidualBlock_atm(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResidualBlock_atm, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.res1 = mutil.ResidualBlock_noBN(gc)
        self.map1 = atmLayer(channel_in)
        self.res2 = mutil.ResidualBlock_noBN(gc)
        self.map2 = atmLayer(channel_in)
        self.res3 = mutil.ResidualBlock_noBN(gc)
        self.map3 = atmLayer(channel_in)
        self.conv2 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1], 0.1)
        else:
            mutil.initialize_weights([self.conv1], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res1(x1) * self.map1(x)
        x3 = self.res2(x2) * self.map2(x)
        x4 = self.res3(x3) * self.map3(x)
        x5 = self.conv2(x4)
        return x5

class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.res1 = mutil.ResidualBlock_noBN(gc)
        self.res2 = mutil.ResidualBlock_noBN(gc)
        self.res3 = mutil.ResidualBlock_noBN(gc)
        self.conv2 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1], 0.1)
        else:
            mutil.initialize_weights([self.conv1], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.conv2(x4)
        return x5
class ResidualNet(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, channel_in, channel_out, init='xavier', nf=64, bias=True):
        super(ResidualNet, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf, channel_out, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)
        # initialization
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        identity = x
        out = self.prelu(self.conv1(x))
        out = self.prelu(self.conv2(out))
        out = self.prelu(self.conv3(out))
        out = self.prelu(self.conv4(out))
        out = self.conv5(out)
        return identity + out
def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'ResNet':
            if init == 'xavier':
                return ResidualNet(channel_in, channel_out, init)
            else:
                return ResidualNet(channel_in, channel_out)
        elif net_structure == 'ResAT2Net':
            if init == 'xavier':
                return ResidualBlock_AT(channel_in, channel_out, init)
            else:
                return ResidualBlock_AT(channel_in, channel_out)
        elif net_structure == 'ResAT2Net_skip':
            if init == 'xavier':
                return ResidualBlock_AT_skip(channel_in, channel_out, init)
            else:
                return ResidualBlock_AT_skip(channel_in, channel_out)
        elif net_structure == 'ResNet_SE':
            if init == 'xavier':
                return ResidualBlock_SE(channel_in, channel_out, init)
            else:
                return ResidualBlock_SE(channel_in, channel_out)
        elif net_structure == 'ResNet_atm':
            if init == 'xavier':
                return ResidualBlock_atm(channel_in, channel_out, init)
            else:
                return ResidualBlock_atm(channel_in, channel_out)
        elif net_structure == 'FBNet':
            if init == 'xavier':
                return FBBlock(channel_in, channel_out, init)
            else:
                return FBBlock(channel_in, channel_out)
        else:
            return None

    return constructor
