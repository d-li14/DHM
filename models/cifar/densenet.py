import torch
import torch.nn as nn
import math

__all__ = ['DenseNet', 'densenet_40_12', 'densenet_100_12', 'densenet_100_24', 'densenet_bc_100_12', 'densenet_bc_250_24', 'densenet_bc_190_40']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, growth_rate=12, dropout_rate=0):
        super(BasicBlock, self).__init__()
        planes = self.expansion * growth_rate
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1, bias=False)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = torch.cat((x, out), 1)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, growth_rate=12, dropout_rate=0):
        super(Bottleneck, self).__init__()
        planes = self.expansion * growth_rate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growth_rate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropout_rate > 0:
           out = self.dropout(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, dropout_rate=0):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.avgpool(out)

        return out


class DenseNet(nn.Module):
    def __init__(self, block, depth, branch_layers, growth_rate=12, compression_factor=1., dropout_rate=0, num_classes=10):
        self.inplanes = growth_rate * 2 if block == Bottleneck else 16
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if block == BasicBlock else (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, n, growth_rate, dropout_rate)
        self.transition1 = self._make_transition(compression_factor, dropout_rate)
        inplanes_head2 = self.inplanes
        self.layer2 = self._make_layer(block, n, growth_rate, dropout_rate)
        self.transition2 = self._make_transition(compression_factor, dropout_rate)
        inplanes_head1 = self.inplanes
        self.layer3 = self._make_layer(block, n, growth_rate, dropout_rate)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        self.inplanes = inplanes_head2
        self.layer2_head2 = self._make_layer(block, branch_layers[0][0], growth_rate, dropout_rate)
        self.transition2_head2 = self._make_transition(compression_factor, dropout_rate)
        self.layer3_head2 = self._make_layer(block, branch_layers[0][1], growth_rate, dropout_rate)
        self.bn_head2 = nn.BatchNorm2d(self.inplanes)
        self.fc_head2 = nn.Linear(self.inplanes, num_classes)

        self.inplanes = inplanes_head1
        growth_rate *= 3
        self.layer3_head1 = self._make_layer(block, branch_layers[1][0], growth_rate, dropout_rate)
        self.bn_head1 = nn.BatchNorm2d(self.inplanes)
        self.fc_head1 = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, growth_rate=12, dropout_rate=0):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, growth_rate, dropout_rate))
            self.inplanes += growth_rate

        return nn.Sequential(*layers)

    def _make_transition(self, compression_factor=1., dropout_rate=0):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes * compression_factor))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, dropout_rate)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = node2 = self.transition1(x)
        x = self.layer2(x)
        x = node1 = self.transition2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x0 = self.fc(x)

        x = self.layer2_head2(node2)
        x = self.transition2_head2(x)
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


def densenet_40_12(**kwargs):
    """
    Constructs a DenseNet {L = 40, k = 12} model.
    """
    return DenseNet(BasicBlock, 40, [[12, 6], [12]], 12, 1, **kwargs)


def densenet_100_12(**kwargs):
    """
    Constructs a DenseNet {L = 100, k = 12} model.
    """
    return DenseNet(BasicBlock, 100, [[16, 16], [32]], 12, 1, **kwargs)


def densenet_100_24(**kwargs):
    """
    Constructs a DenseNet {L = 100, k = 24} model.
    """
    return DenseNet(BasicBlock, 100, [[16, 16], [32]], 24, 1, **kwargs)


def densenet_bc_100_12(**kwargs):
    """
    Constructs a DenseNet-BC {L = 100, k = 12} model.
    """
    return DenseNet(Bottleneck, 100, [[16, 16], [32]], 12, .5, **kwargs)


def densenet_bc_250_24(**kwargs):
    """
    Constructs a DenseNet-BC {L = 250, k = 24} model.
    """
    return DenseNet(Bottleneck, 250, [[22, 22], [41]], 24, .5, **kwargs)


def densenet_bc_190_40(**kwargs):
    """
    Constructs a DenseNet-BC {L = 190, k = 40} model.
    """
    return DenseNet(Bottleneck, 190, [[16, 16], [31]], 40, .5, **kwargs)

