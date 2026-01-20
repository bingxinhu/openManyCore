from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.sound_tracking.utils import get_core_id
import numpy as np


def generate_g1_data(handler, size_y, size_x):
    data = {
        'conv1': {},
        'relu1': {},
        'max_pool': {},
        'pad_127_for_input_g1': {}
    }

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x in [0]:
            px_out = 9
        else:
            px_out = 10
        data['pad_127_for_input_g1'][(core_x, core_y)] = [[127, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                                          [0, 0, 0, 0]] * px_out

    w_range = [(0, 10), (10, 20), (20, 30), (30, 41)]
    w_range_1 = [(0, 10), (9, 20), (19, 30), (29, 41)]
    w_range_2 = [(0, 11), (9, 21), (19, 31), (29, 41)]
    mp_w_range = [(0, 9), (9, 19), (19, 29), (29, 39)]

    # conv1 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l1_input_split_dict, l1_weight_split_dict, l1_bias_split_dict, l1_output_cut_split_dict = {}, {}, {}, {}
    l1_input_split_dict_1, l1_input_split_dict_2 = {}, {}
    l1_input_raw_data = np.array(handler.parameters()['conv1']['input']).astype(np.int8)  # 2048, 7, 7
    l1_weight_raw_data = np.array(handler.parameters()['conv1']['weight']).astype(np.int8)
    l1_bias_raw_data = np.array(handler.parameters()['conv1']['bias']).astype(np.int32)
    l1_output_cut_raw_data = np.array(handler.parameters()['relu1']['output']).astype(np.int8)

    max_pool_output_dict, max_pool_output_dict_with_pad = {}, {}
    max_pool_output_raw_data = np.array(handler.parameters()['maxpool']['output']).astype(np.int8)
    pad_data = np.zeros_like(max_pool_output_raw_data, dtype=np.int8)
    pad_data[0, :, :] = 127
    max_pool_output_raw_data_with_pad = np.concatenate((max_pool_output_raw_data, pad_data), axis=0)

    for core_y, core_x in product(range(size_y), range(size_x)):
        l1_input_split_dict[(core_x, core_y)] = ((0, 8), (0, 257), w_range[core_x])
        l1_weight_split_dict[(core_x, core_y)] = ((0, 16), (0, 8), (0, 3), (0, 3))
        l1_bias_split_dict[(core_x, core_y)] = ((0, 16),)
        l1_output_cut_split_dict[(core_x, core_y)] = ((0, 16), (0, 255), mp_w_range[core_x])

        l1_input_split_dict_1[(core_x, core_y)] = ((0, 8), (0, 257), w_range_1[core_x])
        l1_input_split_dict_2[(core_x, core_y)] = ((0, 8), (0, 257), w_range_2[core_x])

        max_pool_output_dict[(core_x, core_y)] = ((0, 16), (0, 1), mp_w_range[core_x])
        max_pool_output_dict_with_pad[(core_x, core_y)] = ((0, 32), (0, 1), mp_w_range[core_x])
    data['conv1']['input'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    data['conv1']['input1'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['conv1']['input2'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict_2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    data['conv1']['weight'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_weight_raw_data, split_dict=l1_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['conv1']['bias'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_bias_raw_data, split_dict=l1_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['relu1']['output'] = SoundTrackingDataHandler.tensor_split(
        raw_data=l1_output_cut_raw_data, split_dict=l1_output_cut_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    data['max_pool']['output'] = SoundTrackingDataHandler.tensor_split(
        raw_data=max_pool_output_raw_data, split_dict=max_pool_output_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    data['max_pool']['output_with_pad'] = SoundTrackingDataHandler.tensor_split(
        raw_data=max_pool_output_raw_data_with_pad, split_dict=max_pool_output_dict_with_pad,
        data_type=1, alignment=(16, None, None), dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_g1_data(handler, size_y=1, size_x=4)
