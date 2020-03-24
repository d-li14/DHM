from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import math


__all__ = ['ResNet', 'resnet32', 'resnet110', 'resnet1202']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            c_in = x.shape[1]
            b, c_out, h, w = out.shape
            residual = self.downsample(x)
            zero_padding = torch.zeros(b, c_out - c_in, h, w).cuda()
            residual = torch.cat([residual, zero_padding], dim=1)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            c_in = x.shape[1]
            b, c_out, h, w = out.shape
            residual = self.downsample(x)
            zero_padding = torch.zeros(b, c_out - c_in, h, w).cuda()
            residual = torch.cat([residual, zero_padding], dim=1)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, branch_layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        block = BasicBlock

        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        inplanes_head2 = self.inplanes
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        inplanes_head1 = self.inplanes
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.inplanes = inplanes_head2
        self.layer2_head2 = self._make_layer(block, 32, branch_layers[0][0], stride=2)
        self.layer3_head2 = self._make_layer(block, 64, branch_layers[0][1], stride=2)
        self.avgpool_head2 = nn.AvgPool2d(8, stride=1)
        self.fc_head2 = nn.Linear(64 * block.expansion, num_classes)

        self.inplanes = inplanes_head1
        self.layer3_head1 = self._make_layer(block, 128, branch_layers[1][0], stride=2)
        self.avgpool_head1 = nn.AvgPool2d(8, stride=1)
        self.fc_head1 = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = node2 = self.layer1(x)
        x = node1 = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x0 = self.fc(x)

        x = self.layer2_head2(node2)
        x = self.layer3_head2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x2 = self.fc_head2(x)

        x = self.layer3_head1(node1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc_head1(x)

        return x0, x1, x2


def resnet32(**kwargs):
    """
    Constructs a ResNet-32 model.
    """
    return ResNet(32, [[5, 3], [5]], **kwargs)


def resnet110(**kwargs):
    """
    Constructs a ResNet-110 model.
    """
    return ResNet(110, [[9, 9], [18]], **kwargs)


def resnet1202(**kwargs):
    """
    Constructs a ResNet-1202 model.
    """
    return ResNet(1202, [[100, 100], [200]], **kwargs)
