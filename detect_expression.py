'''
Author: Baoyun Peng
Date: 2021-09-18 12:29:58
Description: classify the pose and expression
'''

import cv2
import numpy as np
import torch
import math
import os
import struct
import torch.nn as nn
from mtcnn import MTCNN

from Crypto.Cipher import AES

try:
    from Crypto.Util.Padding import pad, unpad
except ImportError:
    from Crypto.Util.py3compat import bchr, bord
    def pad(data_to_pad, block_size):
        padding_len = block_size-len(data_to_pad)%block_size
        padding = bchr(padding_len)*padding_len
        return data_to_pad + padding
    def unpad(padded_data, block_size):
        pdata_len = len(padded_data)
        if pdata_len % block_size:
            raise ValueError("Input data is not padded")
        padding_len = bord(padded_data[-1])
        if padding_len<1 or padding_len>min(block_size, pdata_len):
            raise ValueError("Padding is incorrect.")
        if padded_data[-padding_len:]!=bchr(padding_len)*padding_len:
            raise ValueError("PKCS#7 padding is incorrect.")
        return padded_data[:-padding_len]

__all__ = ['detect_expression']

def decrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):
    if not out_filename:
        out_filename = in_filename + '.dec'
    with open(in_filename, 'rb') as infile:
        filesize = struct.unpack('<Q', infile.read(8))[0]
        iv = infile.read(16)
        encryptor = AES.new(key, AES.MODE_CBC, iv)
        with open(out_filename, 'wb') as outfile:
            encrypted_filesize = os.path.getsize(in_filename)
            pos = 8 + 16 # the filesize and IV.
            while pos < encrypted_filesize:
                chunk = infile.read(chunksize)
                pos += len(chunk)
                chunk = encryptor.decrypt(chunk)
                if pos == encrypted_filesize:
                    chunk = unpad(chunk, AES.block_size)
                outfile.write(chunk) 

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

expression_dict = {
    '0': '生气',
    '1': '恶心',
    '2': '害怕',
    '3': '高兴',
    '4': '中性',
    '5': '悲伤',
    '6': '吃惊'
}


def detect_expression(img_list):
    expr_model = wideresnet28_2(n_class=7)
    decrypt_file("pby-pose-1234567".encode('utf-8'),"weights/expression.npy.enc", out_filename='/tmp/expression.npy')
    expr_model.load_state_dict(
        torch.load('/tmp/expression.npy', map_location='cpu')['model'])
    expr_model.eval()
    os.system('rm /tmp/expression.npy')
    detector = MTCNN()
    detect_results = {}
    for img_path in img_list:
        image = cv2.imread(img_path.strip())
        detect_results[img_path] = []
        # use mtcnn when mediapipe fails to detect face and pose
        mtcnn_result = detector.detect_faces(image)
        if len(mtcnn_result) == 0:
            continue
        for _idx, _det in enumerate(mtcnn_result):
            lx, ly, h, w = _det['box']
            face_img = image[ly:(ly + w), lx:(lx + h), :]

            face_img = np.transpose(cv2.resize(face_img, (112, 112)),
                                    (2, 0, 1))
            face_img = (face_img / 255.0 - 0.508) / 0.255
            inputs = torch.autograd.Variable(
                torch.from_numpy(face_img[np.newaxis, :, :, :]).float())
            predict_expr = expr_model(inputs)
            probs = torch.nn.functional.softmax(predict_expr, dim=1)
            expr = expression_dict[str(torch.argmax(probs).item())]
            score = torch.max(probs).item()

            detect_results[img_path].append({
                "label": expr,
                "score": score,
                "face_bbox": [lx, ly, lx + h, ly + w]
            })
    return detect_results
