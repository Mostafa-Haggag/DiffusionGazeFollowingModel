import math

import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

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
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, in_channels=3, layers=[3, 4, 6, 3, 2], inplanes=64):
        super(ResNet, self).__init__()
        # layers it is set to [3, 4, 6, 3, 2]
        # inplances = 64
        # bottle neck is a class that we call 
        self.inplanes = inplanes # it can be 3 or 4 it depends

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)#3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#3
        self.layer5 = self._make_layer(block, 256, layers[4], stride=1)#2

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # blocks [3, 4, 6, 3, 2]
        # self.inplaces =3/4
        # block.expansion is equal to 4
        # planes can be 64,128,256,512,256
        # we will always enter in this conditions 
        if stride != 1 or self.inplanes != planes * block.expansion:
            #4,6,3 will enter here
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion# this is changing
        '''
        INTIALLYY inplanes = 3/4
                     PLANES INPLANES    BLOCKS     STRIDE      PLANES
        inplanes = 4 * 64   = 256  -------> 3--------> 1---------> 64  
        inplanes = 4 * 128  = 512  -------> 4--------> 2---------> 128
        inplanes = 4 * 256  = 1024 -------> 6--------> 2---------> 256 
        inplanes = 4 * 512  = 2048 -------> 3--------> 2---------> 512
        inplanes = 4 * 256  = 1024 -------> 2--------> 1---------> 256
        '''
        for _ in range(1, blocks):# we repeate a number of tims this module
            layers.append(block(self.inplanes, planes))
            # I am repeating the same block more than onceee
            

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out