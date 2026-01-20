import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from generator.nsm.nsm import main_net, Cut
import torch
import torch.nn as nn
from generator.mapping_utils.data_handler import DataHandler
from generator.nsm.quantization_config import QuantizationConfig
from collections import OrderedDict


class NSMDataHandler(DataHandler):
    def __init__(self, seed=3, pretrained=False, quantization_en=True):
        super(NSMDataHandler, self).__init__()

        self.qconfig = QuantizationConfig()

        torch.manual_seed(seed)
        self.__model = main_net(state_num=5, input_num=4, hidden_num=100, quantization=quantization_en,
                                in_cut_start=self.qconfig)

        if pretrained:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load('generator/nsm/nsm_parameter.pkl', map_location=device)
            self.__model.load_state_dict(state_dict)
        else:
            for name, module in self.__model.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.mul(127).round().clamp(-128, 127)
                    if module.bias is not None:
                        module.bias.data = module.bias.data.mul(127 * 127).round().clamp(-2 ** 31, 2 ** 31 - 1)

        s = torch.rand((1, 5))
        t = torch.rand((1, 4))
        s = s.mul(127).round().clamp(-128, 127)
        t = t.mul(127).round().clamp(-128, 127)

        _ = self.__model(s, t)
        self.__internal = self.__model.summary

    @property
    def names(self):
        return self.__internal.keys()

    @property
    def parameters(self):
        return self.__internal


if __name__ == '__main__':
    handler = NSMDataHandler()
    print(handler.names)
