from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type, frozen_bn=False):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        if frozen_bn:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_feat(self, x, reture_inter=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if reture_inter:
            return [x2, x3, x4]
        return x4

    def forward(self, x, skip_early=False, forward_feat=False, return_inter=False):
        # only one true
        if forward_feat:
            return self.forward_feat(x, reture_inter=return_inter)
            
        if not skip_early:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            return x

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)

        self.load_state_dict(org_resnet, strict=False)
        print("Initialize resnet from model zoo")


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class UpsampleNet(nn.Module):
    def __init__(self, feat_level=3, in_channels=[512, 1024, 2048], out_channels=256):
        super(UpsampleNet, self).__init__()
        self.feat_level = feat_level
        self.in_chinnels = in_channels
        self.out_channels = out_channels
        self.conv1x1_2048_256 = conv1x1(2048, out_channels)
        self.conv1x1_1024_256 = conv1x1(1024, out_channels)
        self.conv1x1_512_256 = conv1x1(512, out_channels)
        # self.conv1x1_256_512 = conv1x1(256, 256)
        self.sepConv1 = nn.Sequential(
            conv3x3(out_channels, out_channels, 1, out_channels),
            conv1x1(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.sepConv2 = nn.Sequential(
            conv3x3(out_channels, out_channels, 1, out_channels),
            conv1x1(out_channels, out_channels),
            # nn.BatchNorm2d(256),
            # nn.ReLU()
        )
        self.w = nn.Parameter(torch.Tensor(2, 2).fill_(0.5))
        self.eps = 1e-5

    def forward(self, features: List[torch.Tensor]):
        w = F.relu(self.w)
        w = w / (torch.sum(w, dim=0) + self.eps)
        x = self.conv1x1_2048_256(features[2]) # (B, 256, 7, 7)
        x = (w[0, 0] * self.conv1x1_1024_256(features[1]) + w[1, 0] * F.interpolate(x, scale_factor=2, mode='nearest')) / \
            (w[0, 0] + w[1, 0] + self.eps)
        x = self.sepConv1(x) # (B, 256, 14, 14)
        x = (w[0, 1] * self.conv1x1_512_256(features[0]) + w[1, 1] * F.interpolate(x, scale_factor=2, mode='nearest')) / \
            (w[0, 1] + w[1, 1] + self.eps)
        x = self.sepConv2(x) # (B, 256, 28, 28)
        # x = (w[0, 2] * self.conv1x1_256_512(x1) + w[1, 2] * F.interpolate(x, scale_factor=2, mode='nearest')) / \
        #     (w[0, 2] + w[1, 2] + self.eps)
        # x = self.sepConv3(x) # (B, 512, 56, 56)   
        return x

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias