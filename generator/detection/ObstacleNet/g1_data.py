from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g1_data(handler, size_y, size_x):
    data = {}
    data['conv1'] = {}
    data['maxpool1'] = {}
    
    conv1_input_raw_data = np.array(handler.parameters['conv1']['input']).astype(np.int8)
    conv1_weight_raw_data = np.array(handler.parameters['conv1']['weight']).astype(np.int8)
    conv1_bias_raw_data = np.array(handler.parameters['conv1']['bias']).astype(np.int32)
    maxpool1_output_raw_data = np.array(handler.parameters['maxpool1']['output']).astype(np.int8)
    conv1_input_split_dict = {}
    conv1_weight_split_dict = {}
    conv1_bias_split_dict = {}
    maxpool1_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            conv1_input_split_dict[(core_x, core_y)] = ((0, conv1_input_raw_data.shape[0]),
                                                        (core_x * 32, (core_x + 1) * 32),
                                                        (0, conv1_input_raw_data.shape[2]))
        else:
            conv1_input_split_dict[(core_x, core_y)] = ((0, conv1_input_raw_data.shape[0]),
                                                        (core_x * 32, (core_x + 1) * 32 + 1),
                                                        (0, conv1_input_raw_data.shape[2]))
    data['conv1']['input'] = DataHandler.tensor_split(
        raw_data=conv1_input_raw_data, split_dict=conv1_input_split_dict, data_type=1,
        alignment=(None, None, 16), dims=[2, 1, 0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            conv1_weight_split_dict[(core_x, core_y)] = ((0, conv1_weight_raw_data.shape[0] // 2), 
                                                         (0, conv1_weight_raw_data.shape[1]), 
                                                         (0, conv1_weight_raw_data.shape[2]),
                                                         (0, conv1_weight_raw_data.shape[3]))
        else:
            conv1_weight_split_dict[(core_x, core_y)] = ((conv1_weight_raw_data.shape[0] // 2, conv1_weight_raw_data.shape[0]), 
                                                         (0, conv1_weight_raw_data.shape[1]), 
                                                         (0, conv1_weight_raw_data.shape[2]),
                                                         (0, conv1_weight_raw_data.shape[3]))
    data['conv1']['weight'] = DataHandler.tensor_split(
        raw_data=conv1_weight_raw_data, split_dict=conv1_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 3, 2, 1], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            conv1_bias_split_dict[(core_x, core_y)] = ((0, conv1_bias_raw_data.shape[0] // 2), )
        else:
            conv1_bias_split_dict[(core_x, core_y)] = ((conv1_bias_raw_data.shape[0] // 2, conv1_bias_raw_data.shape[0]), )
    data['conv1']['bias'] = DataHandler.tensor_split(
        raw_data=conv1_bias_raw_data, split_dict=conv1_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            if core_x == size_x - 1:
                maxpool1_output_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                                (core_x * 8, (core_x + 1) * 8 - 1),
                                                                (0, maxpool1_output_raw_data.shape[2]))
            else:          
                maxpool1_output_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                                (core_x * 8, (core_x + 1) * 8),
                                                                (0, maxpool1_output_raw_data.shape[2]))
        else:
            if core_x == size_x - 1:
                maxpool1_output_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                                (core_x * 8, (core_x + 1) * 8 - 1),
                                                                (0, maxpool1_output_raw_data.shape[2]))
            else:
                maxpool1_output_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                                (core_x * 8, (core_x + 1) * 8),
                                                                (0, maxpool1_output_raw_data.shape[2]))
    data['maxpool1']['output'] = DataHandler.tensor_split(
        raw_data=maxpool1_output_raw_data, split_dict=maxpool1_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    # 用于测试数据收发
    maxpool1_output1_split_dict = {}
    maxpool1_output2_split_dict = {}
    maxpool1_output3_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            maxpool1_output1_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                             (core_x * size_x, core_x * size_x + 3),
                                                             (0, maxpool1_output_raw_data.shape[2]))
        else:
            maxpool1_output1_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                             (core_x * size_x, core_x * size_x + 3),
                                                             (0, maxpool1_output_raw_data.shape[2]))
    data['maxpool1']['output1'] = DataHandler.tensor_split(
        raw_data=maxpool1_output_raw_data, split_dict=maxpool1_output1_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            maxpool1_output2_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                             (core_x * size_x + 3, core_x * size_x + 6),
                                                             (0, maxpool1_output_raw_data.shape[2]))
        else:
            maxpool1_output2_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                             (core_x * size_x + 3, core_x * size_x + 6),
                                                             (0, maxpool1_output_raw_data.shape[2]))
    data['maxpool1']['output2'] = DataHandler.tensor_split(
        raw_data=maxpool1_output_raw_data, split_dict=maxpool1_output2_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            if core_x == size_x - 1:
                maxpool1_output3_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                                 (core_x * size_x + 6, core_x * size_x + 7),
                                                                 (0, maxpool1_output_raw_data.shape[2]))
            else:
                maxpool1_output3_split_dict[(core_x, core_y)] = ((0, maxpool1_output_raw_data.shape[0] // 2),
                                                                 (core_x * size_x + 6, core_x * size_x + 8),
                                                                 (0, maxpool1_output_raw_data.shape[2]))                
        else:
            if core_x == size_x - 1:
                maxpool1_output3_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                                 (core_x * size_x + 6, core_x * size_x + 7),
                                                                 (0, maxpool1_output_raw_data.shape[2]))
            else:
                maxpool1_output3_split_dict[(core_x, core_y)] = ((maxpool1_output_raw_data.shape[0] // 2, maxpool1_output_raw_data.shape[0]),
                                                                 (core_x * size_x + 6, core_x * size_x + 8),
                                                                 (0, maxpool1_output_raw_data.shape[2]))                
    data['maxpool1']['output3'] = DataHandler.tensor_split(
        raw_data=maxpool1_output_raw_data, split_dict=maxpool1_output3_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g1_data(handler, size_y=2, size_x=8)
