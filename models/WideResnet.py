import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils import model_zoo
import pickle


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=0.01)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)
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
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, dropout=0):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.K = 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16 * self.K, layers[0])
        self.inplanes = self.next_inplanes
        layer2_output_dim = 32 * self.K
        layer3_output_dim = 64 * self.K

        self.layer2 = self._make_layer(block, layer2_output_dim, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_dropout = True if dropout != 0 else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(layer3_output_dim * block.expansion, num_classes)

        self.bn_last = nn.BatchNorm2d(layer3_output_dim, momentum=0.01)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_last(x)
        x = self.relu_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def _resnet(block, layers, n_class, **kwargs):
    model = ResNet(block, layers, num_classes=n_class, **kwargs)
    return model


def wideresnet28_2(n_class, **kwargs):
    return _resnet(BasicBlock, [4, 4, 4], n_class=n_class, **kwargs)


def load_pretrain(model, path, backbone):
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    }
    if backbone == 'resnet18':
        crop_dict = model_zoo.load_url(model_urls['resnet18'], progress=False)
    else:
        pretrained_dict = torch.load(path)['model']

    new_dict = {}
    model_dict = model.state_dict()

    for k, _ in model_dict.items():
        if k in pretrained_dict:
            if k.split('.')[0] != 'linear':
                new_dict[k] = pretrained_dict[k]
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    return model


def main():
    import torch
    import torchvision.models as models
    from torch import nn, optim

    model = wideresnet28_2(7)
    # model = models.resnet50()
    load_pretrain(model, '../pretrain/resnet18', 'resnet18')
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for i in range(10):
        input = torch.randn((64, 3, 112, 112))
        target = torch.ones(64, dtype=torch.long)
        # model = models.resnet18()

        output, l = model(input, input)
        # output = model(input)
        l1 = loss(output[0], target) + l[0] + l[1] + l[2]
        # l1 = loss(output, target)
        optimizer.zero_grad()
        l1.backward()
        # l2.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
