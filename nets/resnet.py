from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import os
from nets.layers import MixStyle#, conv2d, linear, batchnorm

# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlockMeta(nn.Module):
#     expansion: int = 1
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(BasicBlockMeta, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes,affine=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class BottleneckMeta(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

#     expansion: int = 4
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(BottleneckMeta, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width,affine=True)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width,affine=True)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion,affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

class ResNet(nn.Module):
    def __init__(self, block, layers, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        # self.percent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False


    """ Taken from RSC: https://github.com/DeLightCMU/RSC/blob/master/Domain_Generalization/models/resnet.py """
    def forward(self, x, **kwargs):
    # def forward(self, x, gt=None, flag=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)

class ResNetIN(ResNet):
    def __init__(self, block, layers, classes=100):
        super(ResNetIN, self).__init__(block=block, layers=layers, classes=classes)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        # self.percent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,norm_layer=nn.InstanceNorm2d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,norm_layer=nn.InstanceNorm2d))

        return nn.Sequential(*layers)


class ModelParallelResNet50(ResNet):
    def __init__(self, block, layers, classes=100):
        super(ModelParallelResNet50, self).__init__(
            block, layers, classes=classes)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.class_classifier.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.class_classifier(x.view(x.size(0), -1))

class JigsawResNet(ResNet):
    def __init__(self, block, layers, classes=100):
        super(JigsawResNet, self).__init__(block, layers, classes)
        self.jig_classifier = nn.Linear(self.class_classifier.in_features, 31)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x), self.jig_classifier(x)

class MixStyleResNet(ResNet):
    def __init__(self, block, layers, classes=100, mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.1):
        super(MixStyleResNet, self).__init__(block, layers, classes)
        self.mixstyle = None
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='random')
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer2(x)
        if 'layer2' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer3(x)
        if 'layer3' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)

def resnet4(args, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dg_method.lower() == 'jigsaw':
        model = JigsawResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif args.dg_method.lower() == 'mixstyle':
        model = MixStyleResNet(BasicBlock, [1, 1, 1, 1], mixstyle_layers=['layer1', 'layer2', 'layer3'], **kwargs)
    else:
        model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(args, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dg_method.lower() == 'jigsaw':
        model = JigsawResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif args.dg_method.lower() == 'mixstyle':
        model = MixStyleResNet(BasicBlock, [2, 2, 2, 2], mixstyle_layers=['layer1', 'layer2', 'layer3'], **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/research/dept5/mrjiang/Project/DG4FL/pretrained_models'), strict=False)
    if pretrained:
        if os.path.exists('/research/dept5/mrjiang/Project/DG4FL/pretrained_models'):
            print('Use pretrained resnet18')
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/research/dept5/mrjiang/Project/DG4FL/pretrained_models'), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet18IN(args, pretrained=False, **kwargs):
    return ResNetIN(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(args, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dg_method.lower() == 'jigsaw':
        model = JigsawResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif args.dg_method.lower() == 'mixstyle':
        model = MixStyleResNet(Bottleneck, [3, 4, 6, 3], mixstyle_layers=['layer1', 'layer2', 'layer3'], **kwargs)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # model = ModelParallelResNet50(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    if pretrained:
        if os.path.exists('/research/dept5/mrjiang/Project/DG4FL/pretrained_models'):
            print('Use pretrained resnet50')
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],model_dir='/research/dept5/mrjiang/Project/DG4FL/pretrained_models'), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model