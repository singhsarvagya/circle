from __future__ import absolute_import
import torch.nn as nn

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
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
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, in_channels=1, output=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.inplanes = 16
        # self.avgpool_1 = nn.AvgPool2d(5)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, n)
        self.layer2 = self._make_layer(BasicBlock, 32, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, n, stride=2)
        self.avgpool_2 = nn.AvgPool2d(10)
        self.fc = nn.Linear(64*5*5, output)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.avgpool_1(x)
        x = self.conv1(x)  # 1x40x40 -> 16x40x40
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 16x40x40 -> 16x40x40
        x = self.layer2(x)  # 16x40x40 -> 32x20x20
        x = self.layer3(x)  # 16x20x20 -> 64x10x10

        x = self.avgpool_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x
