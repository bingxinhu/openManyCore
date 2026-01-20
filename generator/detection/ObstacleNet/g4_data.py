from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g4_data(handler, size_y, size_x):
    data = {}
    data['conv3'] = {}
    data['maxpool3'] = {}
    
    conv3_input_raw_data = np.array(handler.parameters['conv3']['input']).astype(np.int8)
    conv3_weight_raw_data = np.array(handler.parameters['conv3']['weight']).astype(np.int8)
    conv3_bias_raw_data = np.array(handler.parameters['conv3']['bias']).astype(np.int32)
    maxpool3_output_raw_data = np.array(handler.parameters['maxpool3']['output']).astype(np.int8)
    conv3_input_split_dict = {}
    conv3_weight_split_dict = {}
    conv3_bias_split_dict = {}
    maxpool3_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3_input_split_dict[(core_x, core_y)] = ((0, conv3_input_raw_data.shape[0]),
                                                    (0, conv3_input_raw_data.shape[1]),
                                                    (0, conv3_input_raw_data.shape[2]))
    data['conv3']['input'] = DataHandler.tensor_split(
        raw_data=conv3_input_raw_data, split_dict=conv3_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3_weight_split_dict[(core_x, core_y)] = ((core_x * conv3_weight_raw_data.shape[0] // 4, (core_x + 1) * conv3_weight_raw_data.shape[0] // 4), 
                                                     (0, conv3_weight_raw_data.shape[1]), 
                                                     (0, conv3_weight_raw_data.shape[2]),
                                                     (0, conv3_weight_raw_data.shape[3]))
    data['conv3']['weight'] = DataHandler.tensor_split(
        raw_data=conv3_weight_raw_data, split_dict=conv3_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3_bias_split_dict[(core_x, core_y)] = ((core_x * conv3_bias_raw_data.shape[0] // 4, (core_x + 1) * conv3_bias_raw_data.shape[0] // 4), )
    data['conv3']['bias'] = DataHandler.tensor_split(
        raw_data=conv3_bias_raw_data, split_dict=conv3_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        maxpool3_output_split_dict[(core_x, core_y)] = ((core_x * maxpool3_output_raw_data.shape[0] // 4, (core_x + 1) * maxpool3_output_raw_data.shape[0] // 4),
                                                        (0, maxpool3_output_raw_data.shape[1]),
                                                        (0, maxpool3_output_raw_data.shape[2]))
    data['maxpool3']['output'] = DataHandler.tensor_split(
        raw_data=maxpool3_output_raw_data, split_dict=maxpool3_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g4_data(handler, size_y=1, size_x=4)
