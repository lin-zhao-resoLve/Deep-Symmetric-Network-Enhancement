import torch.nn as nn
from torchvision.models import vgg19


class VGG19(nn.Module):
    def __init__(self, final_layer, pretrain=False):
        super(VGG19, self).__init__()

        self.mean_pixel = (123.68, 116.779, 103.939)
        vgg19_cfg = ['conv_1-1', 'conv_1-2', 'relu_1-2', 'pool_1',
                     'conv_2-1', 'relu_2-1', 'conv_2-2', 'relu_2-2', 'pool_2',
                     'conv_3-1', 'relu_3-1', 'conv_3-2', 'relu_3-2',
                     'conv_3-3', 'relu_3-3', 'conv_3-4', 'relu_3-4', 'pool_3',
                     'conv_4-1', 'relu_4-1', 'conv_4-2', 'relu_4-2',
                     'conv_4-3', 'relu_4-3', 'conv_4-4', 'relu_4-4', 'pool_4',
                     'conv_5-1', 'relu_5-1', 'conv_5-2', 'relu_5-2',
                     'conv_5-3', 'relu_5-3', 'conv_5-4', 'relu_5-4', 'pool_5',
                     ]
        self.prev_relu_idx = []
        layer_setting = {'1': 64, '2': 128, '3': 256, '4': 512, '5': 512}
        layers = []
        in_channel = 3
        for idx, layer_name in enumerate(vgg19_cfg):
            type_name, layer = layer_name.split('_')
            if type_name == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type_name == 'conv':
                layer1, layer2 = layer.split('-')
                out_channel = layer_setting[layer1]
                layers.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
                in_channel = out_channel
            elif type_name == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError('Not support such function {}!'.format(type_name))
            if layer_name == final_layer:
                break
        self.features = nn.ModuleList(layers)

        if pretrain:
            vgg19_state_dict = vgg19(True).state_dict()
            self.features.load_state_dict(vgg19_state_dict, False)

    def forward(self, x):
        res = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.prev_relu_idx:
                res.append(x.clone())
        return res, x


if __name__ == '__main__':
    import torch
    # vgg19_new = VGG19('relu_5-1', ['relu_1-1', 'relu_2-1', 'relu_3-1'], True)
    # x = torch.randn(1, 3, 40, 40)
    # y = vgg19_new(x)
    # print(len(y[0]))
    params = list(VGG19('relu_1-2',True).named_parameters())
    print(params)
    # encoding1 = params[0][1].data
    # encoding2 = params[1][1].data
    print(params[0][1].shape)
