import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from generator.sound_tracking.sound_tracking import NeuralNetwork
import torch
import torch.nn as nn
from generator.mapping_utils.data_handler import DataHandler
from generator.sound_tracking.quantization_config import QuantizationConfig


class SoundTrackingDataHandler(DataHandler):
    def __init__(self, input_channels, output_channels, hidden_size, sequence_length=39, seed=5):
        super(SoundTrackingDataHandler, self).__init__()
        torch.manual_seed(seed)

        self.qconfig = QuantizationConfig(sequence_length=sequence_length)

        self.__model = NeuralNetwork(input_channels=input_channels, output_channels=output_channels,
                                     hidden_size=hidden_size,
                                     in_cut_start_mat=self.qconfig['in_cut_start'], q_one_list=self.qconfig['q_one'],
                                     lut_mat=self.qconfig['lut'], quantization_en=True, conv1_cut=self.qconfig['conv1'],
                                     mlp_cut=self.qconfig['mlp'],
                                     mlp_tanh_d=self.qconfig['mlp_tanh_d'], mlp_tanh_m=self.qconfig['mlp_tanh_m'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load('generator/sound_tracking/quantized_sound_tracking_77.86829599456891.pth', map_location=device)
        self.__model.load_state_dict(state_dict)

        # for name, module in self.__model.named_modules():
        #     if isinstance(module, (nn.Conv2d, nn.Linear)):
        #         module.weight.data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.bias.data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #     elif isinstance(module, GRUCell):
        #         module.in2hid_w[0].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.in2hid_w[1].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.in2hid_w[2].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.in2hid_b[0].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #         module.in2hid_b[1].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #         module.in2hid_b[2].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #         module.hid2hid_w[0].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.hid2hid_w[1].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.hid2hid_w[2].data.mul_(128).floor_().clamp_(min=-128, max=127)
        #         module.hid2hid_b[0].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #         module.hid2hid_b[1].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)
        #         module.hid2hid_b[2].data.mul_(128 * 128).floor_().clamp_(min=-(2**32), max=2**32-1)

        x = torch.randn((1, 8, 257, 41))
        x = (0.1 * x / x.max()).mul(128).floor().clamp(-128, 127)
        _ = self.__model(x)
        self.__internal = self.__model.summary
        self.sequence_length = sequence_length
        self.gru_hidden_size = hidden_size

    def parameters(self):
        return self.__internal

    @property
    def model(self):
        return self.__model


if __name__ == '__main__':
    input_channels = 8
    output_channels = 16
    hidden_size = 128
    para = SoundTrackingDataHandler(input_channels=input_channels, output_channels=output_channels, hidden_size=hidden_size, sequence_length=39)
    xx = 1
