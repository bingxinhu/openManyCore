from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d
import warnings

sys.path.append(os.getcwd())
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
import numpy as np


def generate_g3_data(handler, size_y, size_x):
    g3_data = {}

    g3_data['g3_input'] = {}
    g3_input_raw_data = np.append(np.array(handler.parameters()['mlp']['input'], np.int8),
                                  np.array([[127, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], np.int8))
    g3_input_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        g3_input_split_dict[(core_x, core_y)] = ((0, g3_input_raw_data.shape[0]),)

    g3_data['g3_input'] = SoundTrackingDataHandler.tensor_split(
                            raw_data=g3_input_raw_data,
                            split_dict=g3_input_split_dict,
                            data_type=1, alignment=(16, ), 
                            dims=[0])

    g3_data['g3_weight'] = {}
    g3_weight_raw_data = np.array(handler.parameters()['mlp']['weight'], np.int8)
    g3_bias_raw_data = handler.parameters()['mlp']['bias'] / 127
    if not (g3_bias_raw_data == g3_bias_raw_data.clamp(-128, 127).floor()).all():
        warnings.warn('Bias cannot be divided by 127')
    g3_bias_raw_data = np.array(g3_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
    g3_weight_raw_data = np.concatenate((g3_weight_raw_data, g3_bias_raw_data.reshape((2, 1))), axis=1)
    g3_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        g3_weight_split_dict[(core_x, core_y)] = ((0, g3_weight_raw_data.shape[0]),
                                                  (0, g3_weight_raw_data.shape[1]))

    g3_data['g3_weight'] = SoundTrackingDataHandler.tensor_split(
                            raw_data=g3_weight_raw_data,
                            split_dict=g3_weight_split_dict,
                            data_type=1, alignment=(32, None), 
                            dims=[0, 1])

    g3_data['g3_bias'] = {}
    g3_bias_raw_data = np.array(handler.parameters()['mlp']['bias'], np.int32)
    g3_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        g3_bias_split_dict[(core_x, core_y)] = ((0, g3_bias_raw_data.shape[0]),)

    g3_data['g3_bias'] = SoundTrackingDataHandler.tensor_split(
                            raw_data=g3_bias_raw_data,
                            split_dict=g3_bias_split_dict,
                            data_type=0, alignment=(32, ), 
                            dims=[0])

    g3_data['g3_output'] = {}
    g3_output_raw_data = np.array(handler.parameters()['mlp_tanh']['output'], np.int8)
    g3_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        g3_output_split_dict[(core_x, core_y)] = ((0, g3_output_raw_data.shape[0]),)

    g3_data['g3_output'] = SoundTrackingDataHandler.tensor_split(
                            raw_data=g3_output_raw_data,
                            split_dict=g3_output_split_dict,
                            data_type=1, alignment=(16, ), 
                            dims=[0])

    return g3_data


if __name__ == '__main__':
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    g3_data = generate_g3_data(handler, size_y=1, size_x=1)
    
