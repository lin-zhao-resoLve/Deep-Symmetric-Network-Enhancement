import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = self.conv1_1(X)
        conv1_1 = h
        return conv1_1


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = batch * 255 # [0, 1] -> [0, 255]
    # if opt.vgg_mean:
    #     mean = tensortype(batch.data.size())
    #     mean[:, 0, :, :] = 103.939
    #     mean[:, 1, :, :] = 116.779
    #     mean[:, 2, :, :] = 123.680
    #     batch = batch.sub(Variable(mean))  # subtract mean
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        # if self.opt.no_vgg_instance:
        #     return torch.mean((img_fea - target_fea) ** 2)
        # else:
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)
    # if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
    #     if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
    #         os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
    #     vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
    #     vgg = Vgg16()
    #     for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
    #         dst.data[:] = src
    #     torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg

