import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.getcwd())


def cut(x, in_cut_start: int = 4, en=True, type_out: int = 1):
    """
    result = (x >> (2 * in_cut_start).clip()
    type_out: 0-int32, 1-int8
    """
    if en:
        in_cut_start = in_cut_start
        type_out = type_out
        if type_out == 0:
            qmax, qmin = 0x7fffffff, -0x80000000
        elif type_out == 1:
            qmax, qmin = 0x7f, -0x80
        else:
            raise ValueError
        return x.div(2 ** (2 * in_cut_start)).floor().clamp(min=qmin, max=qmax)
    else:
        return x


class ResidualLayer(nn.Module):
    def __init__(self, inChannels, quantization_en):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels // 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(inChannels // 2, inChannels, kernel_size=3, padding=1)
        self.quantization_en = quantization_en

    def forward(self, x, summary, in_cut_start_dict):
        summary['conv1'] = {}
        summary['conv1']['input'] = x.squeeze(0).detach()
        x1 = self.conv1(x)
        summary['conv1']['weight'] = self.conv1.weight.data
        summary['conv1']['bias'] = self.conv1.bias.data
        summary['conv1']['output'] = x1.squeeze(0).detach()

        summary['conv1_cut'] = {}
        summary['conv1_cut']['input'] = x1.squeeze(0).detach()
        x1 = cut(x1, in_cut_start=in_cut_start_dict['conv1'], en=self.quantization_en)
        summary['conv1_cut']['output'] = x1.squeeze(0).detach()

        summary['conv2'] = {}
        summary['conv2']['input'] = x1.squeeze(0).detach()
        x1 = self.conv2(x1)
        summary['conv2']['weight'] = self.conv2.weight.data
        summary['conv2']['bias'] = self.conv2.bias.data
        summary['conv2']['output'] = x1.squeeze(0).detach()

        summary['conv2_cut'] = {}
        summary['conv2_cut']['input'] = x1.squeeze(0).detach()
        x1 = cut(x1, in_cut_start=in_cut_start_dict['conv2'], en=self.quantization_en)
        summary['conv2_cut']['output'] = x1.squeeze(0).detach()

        summary['add'] = {}
        summary['add']['input1'] = x.squeeze(0).detach()
        summary['add']['input2'] = x1.squeeze(0).detach()
        x = x + x1
        summary['add']['output'] = x.squeeze(0).detach()

        x = F.relu(x)

        summary['add_cut'] = {}
        summary['add_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, in_cut_start=in_cut_start_dict['add'], en=self.quantization_en)
        summary['add_cut']['output'] = x.squeeze(0).detach()

        return x


class ObstacleNet(nn.Module):
    def __init__(self, quantization_en=True, in_cut_start_dict={}):
        super(ObstacleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualLayer(64, quantization_en)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualLayer(128, quantization_en)
        self.res3 = ResidualLayer(128, quantization_en)
        self.conv4 = nn.Conv2d(128, 5 * 5, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()

        self.summary = {}
        self.quantization_en = quantization_en
        self.in_cut_start_dict = in_cut_start_dict

    def forward(self, x):
        assert x.max() < 128 and x.min() >= -128
        self.summary['conv1'] = {}
        self.summary['conv1']['input'] = x.squeeze(0).detach()
        x = self.conv1(x)  # 64, 127, 127
        self.summary['conv1']['weight'] = self.conv1.weight.data
        self.summary['conv1']['bias'] = self.conv1.bias.data
        self.summary['conv1']['output'] = x.squeeze(0).detach()

        self.summary['conv1_cut'] = {}
        self.summary['conv1_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv1'], en=self.quantization_en)
        self.summary['conv1_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu1'] = {}
        self.summary['relu1']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu1']['output'] = x.squeeze(0).detach()

        self.summary['maxpool1'] = {}
        self.summary['maxpool1']['input'] = x.squeeze(0).detach()
        x = self.maxpool1(x)  # 64, 63, 63
        self.summary['maxpool1']['output'] = x.squeeze(0).detach()

        self.summary['res1'] = {}
        self.summary['res1']['input'] = x.squeeze(0).detach()
        x = self.res1(x, self.summary['res1'], self.in_cut_start_dict['res1'])  # 64, 63, 63
        self.summary['res1']['output'] = x.squeeze(0).detach()

        self.summary['conv2'] = {}
        self.summary['conv2']['input'] = x.squeeze(0).detach()
        x = self.conv2(x)  # 128, 31, 31
        self.summary['conv2']['weight'] = self.conv2.weight.data
        self.summary['conv2']['bias'] = self.conv2.bias.data
        self.summary['conv2']['output'] = x.squeeze(0).detach()

        self.summary['conv2_cut'] = {}
        self.summary['conv2_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv2'], en=self.quantization_en)
        self.summary['conv2_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu2'] = {}
        self.summary['relu2']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu2']['output'] = x.squeeze(0).detach()

        self.summary['maxpool2'] = {}
        self.summary['maxpool2']['input'] = x.squeeze(0).detach()
        x = self.maxpool2(x)  # 128, 15, 15
        self.summary['maxpool2']['output'] = x.squeeze(0).detach()

        self.summary['conv3'] = {}
        self.summary['conv3']['input'] = x.squeeze(0).detach()
        x = self.conv3(x)  # 128, 13, 13
        self.summary['conv3']['weight'] = self.conv3.weight.data
        self.summary['conv3']['bias'] = self.conv3.bias.data
        self.summary['conv3']['output'] = x.squeeze(0).detach()

        self.summary['conv3_cut'] = {}
        self.summary['conv3_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv3'], en=self.quantization_en)
        self.summary['conv3_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu3'] = {}
        self.summary['relu3']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu3']['output'] = x.squeeze(0).detach()

        self.summary['maxpool3'] = {}
        self.summary['maxpool3']['input'] = x.squeeze(0).detach()
        x = self.maxpool3(x)  # 128, 6, 6
        self.summary['maxpool3']['output'] = x.squeeze(0).detach()

        self.summary['res2'] = {}
        self.summary['res2']['input'] = x.squeeze(0).detach()
        x = self.res2(x, self.summary['res2'], self.in_cut_start_dict['res2'])  # 128, 6, 6
        self.summary['res2']['output'] = x.squeeze(0).detach()

        self.summary['res3'] = {}
        self.summary['res3']['input'] = x.squeeze(0).detach()
        x = self.res3(x, self.summary['res3'], self.in_cut_start_dict['res3'])  # 128, 6, 6
        self.summary['res3']['output'] = x.squeeze(0).detach()

        self.summary['conv4'] = {}
        self.summary['conv4']['input'] = x.squeeze(0).detach()
        x = self.conv4(x)  # 25, 4, 4
        self.summary['conv4']['weight'] = self.conv4.weight.data
        self.summary['conv4']['bias'] = self.conv4.bias.data
        self.summary['conv4']['output'] = x.squeeze(0).detach()

        self.summary['avgpool'] = {}
        self.summary['avgpool']['input'] = x.squeeze(0).detach()
        x = self.avgpool(x)  # 25, 1, 1
        self.summary['avgpool']['output'] = x.mul(4 * 4).squeeze(0).detach()

        x = x / (128 * 1024)  # 上位机需要实现的操作

        y = x.reshape(x.size(0), 5, 5)  # 5, 5
        y1 = self.act1(y[:, :, 0:4])  # 5, 4
        y2 = self.act2(y[:, :, 4])  # 5
        return y1, y2


class MouseNet(nn.Module):
    def __init__(self, quantization_en=True, in_cut_start_dict={}):
        super(MouseNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualLayer(64, quantization_en)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualLayer(128, quantization_en)
        self.res3 = ResidualLayer(128, quantization_en)
        self.conv4 = nn.Conv2d(128, 5 * 3, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()

        self.summary = {}
        self.quantization_en = quantization_en
        self.in_cut_start_dict = in_cut_start_dict

    def forward(self, x):
        assert x.max() < 128 and x.min() >= -128
        self.summary['conv1'] = {}
        self.summary['conv1']['input'] = x.squeeze(0).detach()
        x = self.conv1(x)  # 64, 127, 127
        self.summary['conv1']['weight'] = self.conv1.weight.data
        self.summary['conv1']['bias'] = self.conv1.bias.data
        self.summary['conv1']['output'] = x.squeeze(0).detach()

        self.summary['conv1_cut'] = {}
        self.summary['conv1_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv1'], en=self.quantization_en)
        self.summary['conv1_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu1'] = {}
        self.summary['relu1']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu1']['output'] = x.squeeze(0).detach()

        self.summary['maxpool1'] = {}
        self.summary['maxpool1']['input'] = x.squeeze(0).detach()
        x = self.maxpool1(x)  # 64, 63, 63
        self.summary['maxpool1']['output'] = x.squeeze(0).detach()

        self.summary['res1'] = {}
        self.summary['res1']['input'] = x.squeeze(0).detach()
        x = self.res1(x, self.summary['res1'], self.in_cut_start_dict['res1'])  # 64, 63, 63
        self.summary['res1']['output'] = x.squeeze(0).detach()

        self.summary['conv2'] = {}
        self.summary['conv2']['input'] = x.squeeze(0).detach()
        x = self.conv2(x)  # 128, 31, 31
        self.summary['conv2']['weight'] = self.conv2.weight.data
        self.summary['conv2']['bias'] = self.conv2.bias.data
        self.summary['conv2']['output'] = x.squeeze(0).detach()

        self.summary['conv2_cut'] = {}
        self.summary['conv2_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv2'], en=self.quantization_en)
        self.summary['conv2_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu2'] = {}
        self.summary['relu2']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu2']['output'] = x.squeeze(0).detach()

        self.summary['maxpool2'] = {}
        self.summary['maxpool2']['input'] = x.squeeze(0).detach()
        x = self.maxpool2(x)  # 128, 15, 15
        self.summary['maxpool2']['output'] = x.squeeze(0).detach()

        self.summary['conv3'] = {}
        self.summary['conv3']['input'] = x.squeeze(0).detach()
        x = self.conv3(x)  # 128, 13, 13
        self.summary['conv3']['weight'] = self.conv3.weight.data
        self.summary['conv3']['bias'] = self.conv3.bias.data
        self.summary['conv3']['output'] = x.squeeze(0).detach()

        self.summary['conv3_cut'] = {}
        self.summary['conv3_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv3'], en=self.quantization_en)
        self.summary['conv3_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu3'] = {}
        self.summary['relu3']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu3']['output'] = x.squeeze(0).detach()

        self.summary['maxpool3'] = {}
        self.summary['maxpool3']['input'] = x.squeeze(0).detach()
        x = self.maxpool3(x)  # 128, 6, 6
        self.summary['maxpool3']['output'] = x.squeeze(0).detach()

        self.summary['res2'] = {}
        self.summary['res2']['input'] = x.squeeze(0).detach()
        x = self.res2(x, self.summary['res2'], self.in_cut_start_dict['res2'])  # 128, 6, 6
        self.summary['res2']['output'] = x.squeeze(0).detach()

        self.summary['res3'] = {}
        self.summary['res3']['input'] = x.squeeze(0).detach()
        x = self.res3(x, self.summary['res3'], self.in_cut_start_dict['res3'])  # 128, 6, 6
        self.summary['res3']['output'] = x.squeeze(0).detach()

        self.summary['conv4'] = {}
        self.summary['conv4']['input'] = x.squeeze(0).detach()
        x = self.conv4(x)  # 15, 4, 4
        self.summary['conv4']['weight'] = self.conv4.weight.data
        self.summary['conv4']['bias'] = self.conv4.bias.data
        self.summary['conv4']['output'] = x.squeeze(0).detach()

        self.summary['avgpool'] = {}
        self.summary['avgpool']['input'] = x.squeeze(0).detach()
        x = self.avgpool(x)  # 15, 1, 1
        self.summary['avgpool']['output'] = x.mul(4 * 4).squeeze(0).detach()

        x = x / (128 * 1024)  # 上位机需要实现的操作

        y = x.reshape(x.size(0), 3, 5)  # 3, 5
        y1 = self.act1(y[:, :, 0:4])  # 3, 4
        y2 = self.act2(y[:, :, 4])  # 3
        return y1, y2


class SDNet(nn.Module):
    def __init__(self, quantization_en=True, in_cut_start_dict={}):
        super(SDNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualLayer(64, quantization_en)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualLayer(128, quantization_en)
        self.res3 = ResidualLayer(128, quantization_en)
        self.conv4 = nn.Conv2d(128, 4, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()

        self.summary = {}
        self.quantization_en = quantization_en
        self.in_cut_start_dict = in_cut_start_dict

    def forward(self, x):
        assert x.max() < 128 and x.min() >= -128
        self.summary['conv1'] = {}
        self.summary['conv1']['input'] = x.squeeze(0).detach()
        x = self.conv1(x)  # 64, 127, 127
        self.summary['conv1']['weight'] = self.conv1.weight.data
        self.summary['conv1']['bias'] = self.conv1.bias.data
        self.summary['conv1']['output'] = x.squeeze(0).detach()

        self.summary['conv1_cut'] = {}
        self.summary['conv1_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv1'], en=self.quantization_en)
        self.summary['conv1_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu1'] = {}
        self.summary['relu1']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu1']['output'] = x.squeeze(0).detach()

        self.summary['maxpool1'] = {}
        self.summary['maxpool1']['input'] = x.squeeze(0).detach()
        x = self.maxpool1(x)  # 64, 63, 63
        self.summary['maxpool1']['output'] = x.squeeze(0).detach()

        self.summary['res1'] = {}
        self.summary['res1']['input'] = x.squeeze(0).detach()
        x = self.res1(x, self.summary['res1'], self.in_cut_start_dict['res1'])  # 64, 63, 63
        self.summary['res1']['output'] = x.squeeze(0).detach()

        self.summary['conv2'] = {}
        self.summary['conv2']['input'] = x.squeeze(0).detach()
        x = self.conv2(x)  # 128, 31, 31
        self.summary['conv2']['weight'] = self.conv2.weight.data
        self.summary['conv2']['bias'] = self.conv2.bias.data
        self.summary['conv2']['output'] = x.squeeze(0).detach()

        self.summary['conv2_cut'] = {}
        self.summary['conv2_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv2'], en=self.quantization_en)
        self.summary['conv2_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu2'] = {}
        self.summary['relu2']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu2']['output'] = x.squeeze(0).detach()

        self.summary['maxpool2'] = {}
        self.summary['maxpool2']['input'] = x.squeeze(0).detach()
        x = self.maxpool2(x)  # 128, 15, 15
        self.summary['maxpool2']['output'] = x.squeeze(0).detach()

        self.summary['conv3'] = {}
        self.summary['conv3']['input'] = x.squeeze(0).detach()
        x = self.conv3(x)  # 128, 13, 13
        self.summary['conv3']['weight'] = self.conv3.weight.data
        self.summary['conv3']['bias'] = self.conv3.bias.data
        self.summary['conv3']['output'] = x.squeeze(0).detach()

        self.summary['conv3_cut'] = {}
        self.summary['conv3_cut']['input'] = x.squeeze(0).detach()
        x = cut(x, self.in_cut_start_dict['conv3'], en=self.quantization_en)
        self.summary['conv3_cut']['output'] = x.squeeze(0).detach()

        self.summary['relu3'] = {}
        self.summary['relu3']['input'] = x.squeeze(0).detach()
        x = self.relu(x)
        self.summary['relu3']['output'] = x.squeeze(0).detach()

        self.summary['maxpool3'] = {}
        self.summary['maxpool3']['input'] = x.squeeze(0).detach()
        x = self.maxpool3(x)  # 128, 6, 6
        self.summary['maxpool3']['output'] = x.squeeze(0).detach()

        self.summary['res2'] = {}
        self.summary['res2']['input'] = x.squeeze(0).detach()
        x = self.res2(x, self.summary['res2'], self.in_cut_start_dict['res2'])  # 128, 6, 6
        self.summary['res2']['output'] = x.squeeze(0).detach()

        self.summary['res3'] = {}
        self.summary['res3']['input'] = x.squeeze(0).detach()
        x = self.res3(x, self.summary['res3'], self.in_cut_start_dict['res3'])  # 128, 6, 6
        self.summary['res3']['output'] = x.squeeze(0).detach()

        self.summary['conv4'] = {}
        self.summary['conv4']['input'] = x.squeeze(0).detach()
        x = self.conv4(x)  # 15, 4, 4
        self.summary['conv4']['weight'] = self.conv4.weight.data
        self.summary['conv4']['bias'] = self.conv4.bias.data
        self.summary['conv4']['output'] = x.squeeze(0).detach()

        self.summary['avgpool'] = {}
        self.summary['avgpool']['input'] = x.squeeze(0).detach()
        x = (x * 16).double()
        x = self.avgpool(x)  # 4, 1, 1
        self.summary['avgpool']['output'] = x.squeeze(0).detach()

        # x = x / (128 * 1024)  # 上位机需要实现的操作
        #
        # y = x.reshape(x.size(0), 3, 5)  # 3, 5
        # y1 = self.act1(y[:, :, 0:4])  # 3, 4
        # y2 = self.act2(y[:, :, 4])  # 3
        return x


if __name__ == '__main__':
    from generator.detection.quantization_config import QuantizationConfig

    x = torch.randn((1, 3, 256, 256))
    x = x.mul(128).round().clamp(-128, 127)

    qconfig = QuantizationConfig(name='obstacle')
    in_cut_start_dict = qconfig['in_cut_start']
    model = ObstacleNet(quantization_en=True, in_cut_start_dict=in_cut_start_dict)
    for name, module in model.named_modules():
        if name == 'conv1':
            module.weight.data = module.weight.data.mul(64).round().clamp(-128, 127)
            module.bias.data = module.bias.data.mul(64 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)
        else:
            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.mul(1024).round().clamp(-128, 127)
                module.bias.data = module.bias.data.mul(1024 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)
    y = model(x)
    print(y)

    qconfig = QuantizationConfig(name='mouse')
    in_cut_start_dict = qconfig['in_cut_start']
    model = MouseNet(quantization_en=True, in_cut_start_dict=in_cut_start_dict)
    for name, module in model.named_modules():
        if name == 'conv1':
            module.weight.data = module.weight.data.mul(64).round().clamp(-128, 127)
            module.bias.data = module.bias.data.mul(64 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)
        else:
            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.mul(1024).round().clamp(-128, 127)
                module.bias.data = module.bias.data.mul(1024 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)
    y = model(x)
    print(y)
