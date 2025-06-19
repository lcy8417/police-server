from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

def resnet_block_dilation(block, dilation_0=None, dilation=2):
    for i in range(0, len(block)):
        # The first sub-block, containing downsample layer and stride (2,2)
        if i == 0:
            for (name, layer) in block[i].named_children():
                if 'downsample' in name:
                    layer[0].stride = (1, 1)
                if 'conv' in name:
                    if layer.stride == (2, 2):
                        layer.stride = (1, 1)
                        if dilation_0 and layer.kernel_size == (3, 3):
                            layer.dilation = (dilation_0, dilation_0)
                            layer.padding = (dilation_0, dilation_0)
        # The others sub-block, containing only simple conv2d
        else:
            for (name, layer) in block[i].named_children():
                if 'conv' in name:
                    if layer.kernel_size == (3, 3):
                        layer.dilation = (dilation, dilation)
                        layer.padding = (dilation, dilation)
    return block

# Kaiming weight initialization
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def extract_features_from_e2e(model):
    state_dict_features = OrderedDict()
    for key, value in model['state_dict'].items():
        if key.startswith('features'):
            state_dict_features[key[9:]] = value
    return state_dict_features


class ResNet(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool):
        super(ResNet, self).__init__()
        net_in = getattr(torchvision.models, name)(weights=True)
        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class ResNet_STAGE45(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool):
        super(ResNet_STAGE45, self).__init__()
        net_in = getattr(torchvision.models, name)(weights=True)
        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x4)
        return x4, x5
