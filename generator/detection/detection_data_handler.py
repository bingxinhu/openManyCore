import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from generator.detection.detection import ObstacleNet
from generator.detection.detection import MouseNet, SDNet
import torch
import torch.nn as nn
from generator.mapping_utils.data_handler import DataHandler
from generator.detection.quantization_config import QuantizationConfig
import cv2


class DetectionDataHandler(DataHandler):
    # img = cv2.imread('./temp/picLoopCheck/picRecorded1.png', cv2.IMREAD_UNCHANGED)
    # x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    # input = x
    input = None

    def __init__(self, seed=15, name='obstacle', pretrained=False, quantization_en=True):
        super(DetectionDataHandler, self).__init__()

        self.qconfig = QuantizationConfig(name=name)

        if name == 'obstacle':
            torch.manual_seed(seed)
            self.__model = ObstacleNet(quantization_en=quantization_en, in_cut_start_dict=self.qconfig['in_cut_start'])
        elif name == 'mouse':
            torch.manual_seed(seed + 1)
            self.__model = MouseNet(quantization_en=quantization_en, in_cut_start_dict=self.qconfig['in_cut_start'])
        elif name == 'sd':
            torch.manual_seed(seed + 2)
            self.__model = SDNet(quantization_en=quantization_en, in_cut_start_dict=self.qconfig['in_cut_start'])
        else:
            raise ValueError

        if pretrained:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if name == 'obstacle':
                state_dict = torch.load('generator/detection/Aware_Quan_obstacle_meanPr=94.90831756591797.pth',
                                        map_location=device)
            elif name == 'mouse':
                state_dict = torch.load(
                    # 'generator/detection/Aware_Quan_Jerry_Pr=99.99524688720703F1=97.66998971798525.pth',
                    'generator/detection/NewAware_Quan_Jerry_Pr=96.55011749267578.pth',
                    map_location=device)
            elif name == 'sd':
                state_dict = torch.load(
                    # 'generator/detection/quantized_SD.pth',
                    'generator/detection/quantized_SD_acc90.909_epoch50.pth',
                    map_location=device)
            else:
                raise ValueError
            self.__model.load_state_dict(state_dict)
        else:
            for name, module in self.__model.named_modules():
                if name == 'conv1':
                    module.weight.data = module.weight.data.mul(64).round().clamp(-128, 127)
                    module.bias.data = module.bias.data.mul(64 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)
                else:
                    if isinstance(module, nn.Conv2d):
                        module.weight.data = module.weight.data.mul(1024).round().clamp(-128, 127)
                        module.bias.data = module.bias.data.mul(1024 * 128).round().clamp(-2 ** 31, 2 ** 31 - 1)

        # inference
        if DetectionDataHandler.input is None:
            torch.manual_seed(0)
            x = torch.randn((1, 3, 256, 256))
            DetectionDataHandler.input = x.mul(128).round().clamp(-128, 127)
        y = self.__model(DetectionDataHandler.input)
        print(y)
        self.__internal = self.__model.summary

    @property
    def names(self):
        return self.__internal.keys()

    @property
    def parameters(self):
        return self.__internal

    @property
    def model(self):
        return self.__model


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle')
    print(handler.names)
