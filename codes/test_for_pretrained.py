import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.model.vgg16 import Vgg16
import os
import cv2
import torchvision.transforms as TF
import options.options as option
import utils.util as util
import torch

vgg = Vgg16()
vgg.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'models/modules/model/', 'vgg16.weight')))
params = list(vgg.named_parameters())
for i in range(len(params)):
    print(params[i][0])
encoding1 = params[0][1].data
encoding2 = params[2][1].data
encoding3 = params[4][1].data
bias1 = params[1][1].data
bias2 = params[3][1].data
print(encoding2.shape)

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

        self.haar_weights3 = encoding3
        self.haar_weights3 = nn.Parameter(self.haar_weights3)
        self.haar_weights3.requires_grad = False

    def forward(self, x):
        out = F.conv2d(x, torch.randn([64, 3, 3, 3]), bias=None, stride=1, padding=1, groups=1)
        out = F.conv2d(out, torch.randn([64, 64, 3, 3]), bias=None, stride=1, padding=1, groups=1)
        # out = F.conv2d(out, self.haar_weights3, bias=None, stride=1, padding=1, groups=1)
        # out = F.conv_transpose2d(out, self.haar_weights3, bias=None, stride=1, padding=1, groups=1)
        out = F.conv_transpose2d(out, torch.randn([64, 64, 3, 3]), bias=None, stride=1, padding=1, groups=1)
        out = F.conv_transpose2d(out, torch.randn([64, 3, 3, 3]), bias=None, stride=1, padding=1, groups=1)
        return out


model = Upsampling(3)

low_im = cv2.imread('/home/lin-zhao/C_4500_500/ceshi/000014.jpg', cv2.IMREAD_COLOR)
low_im = low_im[:, :, [2, 1, 0]]
# low_im = np.expand_dims(low_im[:, :, [2, 1, 0]],axis=0)
input = TF.ToTensor()(low_im.copy())
input = input.unsqueeze(0)
output = model(input)
output_img = util.tensor2img(output)
util.save_img(output_img, '/home/lin-zhao/C_4500_500/ceshi/000014_out_m.jpg')

