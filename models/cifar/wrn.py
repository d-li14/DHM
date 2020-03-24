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


__all__ = ['WideResNet', 'wrn_16_8', 'wrn_28_10']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dropout_rate=0, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.equalInOut = inplanes == planes
        self.shortcut = (not self.equalInOut) and nn.Conv2d(inplanes, planes, 1, stride, bias=False) or None

    def forward(self, x):
        if self.equalInOut:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            x = self.bn1(x)
            x = self.relu(x)

        out = self.conv1(out if self.equalInOut else x)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            x = self.shortcut(x)
        out += x

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, branch_layers, widening_factor=1, dropout_rate=0, num_classes=10):
        self.inplanes = 16
        super(WideResNet, self).__init__()
        block = BasicBlock

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6

        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16 * widening_factor, n, dropout_rate)
        inplanes_head2 = self.inplanes
        self.layer2 = self._make_layer(block, 32 * widening_factor, n, dropout_rate, stride=2)
        inplanes_head1 = self.inplanes
        self.layer3 = self._make_layer(block, 64 * widening_factor, n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(64 * widening_factor * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * widening_factor * block.expansion, num_classes)

        self.inplanes = inplanes_head2
        self.layer2_head2 = self._make_layer(block, 32 * widening_factor, branch_layers[0][0], dropout_rate, stride=2)
        self.layer3_head2 = self._make_layer(block, 64 * widening_factor, branch_layers[0][1], dropout_rate, stride=2)
        self.bn_head2 = nn.BatchNorm2d(64 * widening_factor * block.expansion)
        self.fc_head2 = nn.Linear(64 * widening_factor * block.expansion, num_classes)

        self.inplanes = inplanes_head1
        self.layer3_head1 = self._make_layer(block, 128 * widening_factor, branch_layers[1][0], dropout_rate, stride=2)
        self.bn_head1 = nn.BatchNorm2d(128 * widening_factor * block.expansion)
        self.fc_head1 = nn.Linear(128 * widening_factor * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, dropout_rate=0, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, dropout_rate, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = node2 = self.layer1(x)
        x = node1 = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x0 = self.fc(x)

        x = self.layer2_head2(node2)
        x = self.layer3_head2(x)
        x = self.bn_head2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x2 = self.fc_head2(x)

        x = self.layer3_head1(node1)
        x = self.bn_head1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc_head1(x)

        return x0, x1, x2


def wrn_16_8(**kwargs):
    """
    Constructs a WRN-16-8 model.
    """
    return WideResNet(16, [[2, 1], [2]], 8, **kwargs)


def wrn_28_10(**kwargs):
    """
    Constructs a WRN-28-10 model.
    """
    return WideResNet(28, [[4, 2], [4]], 10, **kwargs)
