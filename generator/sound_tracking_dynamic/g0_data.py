from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler
import numpy as np


def generate_g0_data(handler, size_y, size_x):
    data = {
        'fpga': {}
    }

    w_range = [(0, 10), (10, 20), (20, 30), (30, 41)]
    h_range = [(0, 64), (64, 128), (128, 192), (192, 257)]

    # conv1 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l1_input_split_dict_0, l1_input_split_dict_1, l1_input_split_dict_2, l1_input_split_dict_3 = {}, {}, {}, {}
    l1_input_raw_data = np.array(handler.parameters()['conv1']['input']).astype(np.int8)

    for core_y, core_x in product(range(size_y), range(size_x)):
        l1_input_split_dict_0[(core_x, core_y)] = ((0, 8), h_range[0], w_range[core_x])
        l1_input_split_dict_1[(core_x, core_y)] = ((0, 8), h_range[1], w_range[core_x])
        l1_input_split_dict_2[(core_x, core_y)] = ((0, 8), h_range[2], w_range[core_x])
        l1_input_split_dict_3[(core_x, core_y)] = ((0, 8), h_range[3], w_range[core_x])
    data['fpga']['input1'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_0, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['fpga']['input2'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['fpga']['input3'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['fpga']['input4'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_3, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    return data
