from itertools import product
import sys
import os
from numpy import core
from numpy.core.fromnumeric import size
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g2_data(handler, size_y, size_x):
    data = {}
    data['res1'] = {}
    data['res1']['conv1'] = {}
    data['res1']['conv2'] = {}
    data['res1']['add'] = {}

    
    res1_conv1_input_raw_data = np.array(handler.parameters['res1']['conv1']['input']).astype(np.int8)
    res1_conv1_weight_raw_data = np.array(handler.parameters['res1']['conv1']['weight']).astype(np.int8)
    res1_conv1_bias_raw_data = np.array(handler.parameters['res1']['conv1']['bias']).astype(np.int32)
    res1_conv1_output_raw_data = np.array(handler.parameters['res1']['conv1_cut']['output']).astype(np.int8)
    res1_conv1_input_split_dict = {}
    res1_conv1_weight_split_dict = {}
    res1_conv1_bias_split_dict = {}
    res1_conv1_output_split_dict = {}

    res1_conv1_input1_split_dict = {}
    res1_conv1_input2_split_dict = {}
    res1_conv1_input3_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv1_input1_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                          (core_x * size_x, core_x * size_x + 3),
                                                          (0, res1_conv1_input_raw_data.shape[2]))
    data['res1']['conv1']['input1'] = DataHandler.tensor_split(
        raw_data=res1_conv1_input_raw_data, split_dict=res1_conv1_input1_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv1_input2_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                          (core_x * size_x + 3, core_x * size_x + 6),
                                                          (0, res1_conv1_input_raw_data.shape[2]))
    data['res1']['conv1']['input2'] = DataHandler.tensor_split(
        raw_data=res1_conv1_input_raw_data, split_dict=res1_conv1_input2_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_conv1_input3_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                              (core_x * size_x + 6, core_x * size_x + 7),
                                                              (0, res1_conv1_input_raw_data.shape[2]))
        else:
            res1_conv1_input3_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                              (core_x * size_x + 6, core_x * size_x + 8),
                                                              (0, res1_conv1_input_raw_data.shape[2]))
    data['res1']['conv1']['input3'] = DataHandler.tensor_split(
        raw_data=res1_conv1_input_raw_data, split_dict=res1_conv1_input3_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_conv1_input_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                             (core_x * 8, (core_x + 1) * 8 - 1),
                                                             (0, res1_conv1_input_raw_data.shape[2]))
        else:
            res1_conv1_input_split_dict[(core_x, core_y)] = ((0, res1_conv1_input_raw_data.shape[0]),
                                                             (core_x * 8, (core_x + 1) * 8),
                                                             (0, res1_conv1_input_raw_data.shape[2]))
    data['res1']['conv1']['input'] = DataHandler.tensor_split(
        raw_data=res1_conv1_input_raw_data, split_dict=res1_conv1_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv1_weight_split_dict[(core_x, core_y)] = ((0, res1_conv1_weight_raw_data.shape[0]), 
                                                          (0, res1_conv1_weight_raw_data.shape[1]), 
                                                          (0, res1_conv1_weight_raw_data.shape[2]),
                                                          (0, res1_conv1_weight_raw_data.shape[3]))
    data['res1']['conv1']['weight'] = DataHandler.tensor_split(
        raw_data=res1_conv1_weight_raw_data, split_dict=res1_conv1_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv1_bias_split_dict[(core_x, core_y)] = ((0, res1_conv1_bias_raw_data.shape[0]), )
    data['res1']['conv1']['bias'] = DataHandler.tensor_split(
        raw_data=res1_conv1_bias_raw_data, split_dict=res1_conv1_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_conv1_output_split_dict[(core_x, core_y)] = ((0, res1_conv1_output_raw_data.shape[0]),
                                                              (core_x * 8, (core_x + 1) * 8 - 1),
                                                              (0, res1_conv1_output_raw_data.shape[2]))
        else:
            res1_conv1_output_split_dict[(core_x, core_y)] = ((0, res1_conv1_output_raw_data.shape[0]),
                                                              (core_x * 8, (core_x + 1) * 8),
                                                              (0, res1_conv1_output_raw_data.shape[2]))
    data['res1']['conv1']['output'] = DataHandler.tensor_split(
        raw_data=res1_conv1_output_raw_data, split_dict=res1_conv1_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res1_conv2_weight_raw_data = np.array(handler.parameters['res1']['conv2']['weight']).astype(np.int8)
    res1_conv2_bias_raw_data = np.array(handler.parameters['res1']['conv2']['bias']).astype(np.int32)
    res1_conv2_output_raw_data = np.array(handler.parameters['res1']['conv2_cut']['output']).astype(np.int8)
    res1_conv2_weight_split_dict = {}
    res1_conv2_bias_split_dict = {}
    res1_conv2_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv2_weight_split_dict[(core_x, core_y)] = ((0, res1_conv2_weight_raw_data.shape[0]), 
                                                          (0, res1_conv2_weight_raw_data.shape[1]), 
                                                          (0, res1_conv2_weight_raw_data.shape[2]),
                                                          (0, res1_conv2_weight_raw_data.shape[3]))
    data['res1']['conv2']['weight'] = DataHandler.tensor_split(
        raw_data=res1_conv2_weight_raw_data, split_dict=res1_conv2_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_conv2_bias_split_dict[(core_x, core_y)] = ((0, res1_conv2_bias_raw_data.shape[0]), )
    data['res1']['conv2']['bias'] = DataHandler.tensor_split(
        raw_data=res1_conv2_bias_raw_data, split_dict=res1_conv2_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_conv2_output_split_dict[(core_x, core_y)] = ((0, res1_conv2_output_raw_data.shape[0]),
                                                              (core_x * 8, (core_x + 1) * 8 - 1),
                                                              (0, res1_conv2_output_raw_data.shape[2]))
        else:
            res1_conv2_output_split_dict[(core_x, core_y)] = ((0, res1_conv2_output_raw_data.shape[0]),
                                                              (core_x * 8, (core_x + 1) * 8),
                                                              (0, res1_conv2_output_raw_data.shape[2]))
    data['res1']['conv2']['output'] = DataHandler.tensor_split(
        raw_data=res1_conv2_output_raw_data, split_dict=res1_conv2_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res1_add_output_raw_data = np.array(handler.parameters['res1']['add_cut']['output']).astype(np.int8)
    res1_add_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_add_output_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                            (core_x * 8, (core_x + 1) * 8 - 1),
                                                            (0, res1_add_output_raw_data.shape[2]))
        else:
            res1_add_output_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                            (core_x * 8, (core_x + 1) * 8),
                                                            (0, res1_add_output_raw_data.shape[2]))
    data['res1']['add']['output'] = DataHandler.tensor_split(
        raw_data=res1_add_output_raw_data, split_dict=res1_add_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res1_add_output1_split_dict = {}
    res1_add_output2_split_dict = {}
    res1_add_output3_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_add_output1_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                         (core_x * size_x, core_x * size_x + 3),
                                                         (0, res1_add_output_raw_data.shape[2]))
    data['res1']['add']['output1'] = DataHandler.tensor_split(
        raw_data=res1_add_output_raw_data, split_dict=res1_add_output1_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res1_add_output2_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                         (core_x * size_x + 3, core_x * size_x + 6),
                                                         (0, res1_add_output_raw_data.shape[2]))
    data['res1']['add']['output2'] = DataHandler.tensor_split(
        raw_data=res1_add_output_raw_data, split_dict=res1_add_output2_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1:
            res1_add_output3_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                             (core_x * size_x + 6, core_x * size_x + 7),
                                                             (0, res1_add_output_raw_data.shape[2]))
        else:
            res1_add_output3_split_dict[(core_x, core_y)] = ((0, res1_add_output_raw_data.shape[0]),
                                                             (core_x * size_x + 6, core_x * size_x + 8),
                                                             (0, res1_add_output_raw_data.shape[2]))
    data['res1']['add']['output3'] = DataHandler.tensor_split(
        raw_data=res1_add_output_raw_data, split_dict=res1_add_output3_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g2_data(handler, size_y=1, size_x=8)
