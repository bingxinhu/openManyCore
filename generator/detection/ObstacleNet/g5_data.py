from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g5_data(handler, size_y, size_x):
    data = {}
    data['res2'] = {}
    data['res2']['conv1'] = {}
    data['res2']['conv2'] = {}
    data['res2']['add'] = {}
    data['res3'] = {}
    data['res3']['conv1'] = {}
    data['res3']['conv2'] = {}
    data['res3']['add'] = {}

    
    res2_conv1_input_raw_data = np.array(handler.parameters['res2']['conv1']['input']).astype(np.int8)
    res2_conv1_weight_raw_data = np.array(handler.parameters['res2']['conv1']['weight']).astype(np.int8)
    res2_conv1_bias_raw_data = np.array(handler.parameters['res2']['conv1']['bias']).astype(np.int32)
    res2_conv1_output_raw_data = np.array(handler.parameters['res2']['conv1_cut']['output']).astype(np.int8)
    res2_conv1_input_split_dict = {}
    res2_conv1_weight_split_dict = {}
    res2_conv1_bias_split_dict = {}
    res2_conv1_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        res2_conv1_input_split_dict[(core_x, core_y)] = ((0, res2_conv1_input_raw_data.shape[0]),
                                                         (0, res2_conv1_input_raw_data.shape[1]),
                                                         (0, res2_conv1_input_raw_data.shape[2]))
    data['res2']['conv1']['input'] = DataHandler.tensor_split(
        raw_data=res2_conv1_input_raw_data, split_dict=res2_conv1_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res2_conv1_weight_split_dict[(core_x, core_y)] = ((0, res2_conv1_weight_raw_data.shape[0]), 
                                                          (0, res2_conv1_weight_raw_data.shape[1]), 
                                                          (0, res2_conv1_weight_raw_data.shape[2]),
                                                          (0, res2_conv1_weight_raw_data.shape[3]))
    data['res2']['conv1']['weight'] = DataHandler.tensor_split(
        raw_data=res2_conv1_weight_raw_data, split_dict=res2_conv1_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        res2_conv1_bias_split_dict[(core_x, core_y)] = ((0, res2_conv1_bias_raw_data.shape[0]), )
    data['res2']['conv1']['bias'] = DataHandler.tensor_split(
        raw_data=res2_conv1_bias_raw_data, split_dict=res2_conv1_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res2_conv1_output_split_dict[(core_x, core_y)] = ((0, res2_conv1_output_raw_data.shape[0]),
                                                          (0, res2_conv1_output_raw_data.shape[1]),
                                                          (0, res2_conv1_output_raw_data.shape[2]))
    data['res2']['conv1']['output'] = DataHandler.tensor_split(
        raw_data=res2_conv1_output_raw_data, split_dict=res2_conv1_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res2_conv2_weight_raw_data = np.array(handler.parameters['res2']['conv2']['weight']).astype(np.int8)
    res2_conv2_bias_raw_data = np.array(handler.parameters['res2']['conv2']['bias']).astype(np.int32)
    res2_conv2_output_raw_data = np.array(handler.parameters['res2']['conv2_cut']['output']).astype(np.int8)
    res2_conv2_weight_split_dict = {}
    res2_conv2_bias_split_dict = {}
    res2_conv2_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res2_conv2_weight_split_dict[(core_x, core_y)] = ((0, res2_conv2_weight_raw_data.shape[0] // 2), 
                                                              (0, res2_conv2_weight_raw_data.shape[1]), 
                                                              (0, res2_conv2_weight_raw_data.shape[2]),
                                                              (0, res2_conv2_weight_raw_data.shape[3]))
        else:
            res2_conv2_weight_split_dict[(core_x, core_y)] = ((res2_conv2_weight_raw_data.shape[0] // 2, res2_conv2_weight_raw_data.shape[0]), 
                                                              (0, res2_conv2_weight_raw_data.shape[1]), 
                                                              (0, res2_conv2_weight_raw_data.shape[2]),
                                                              (0, res2_conv2_weight_raw_data.shape[3]))
    data['res2']['conv2']['weight'] = DataHandler.tensor_split(
        raw_data=res2_conv2_weight_raw_data, split_dict=res2_conv2_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res2_conv2_bias_split_dict[(core_x, core_y)] = ((0, res2_conv2_bias_raw_data.shape[0] // 2), )
        else:
            res2_conv2_bias_split_dict[(core_x, core_y)] = ((res2_conv2_bias_raw_data.shape[0] // 2, res2_conv2_bias_raw_data.shape[0]), )
    data['res2']['conv2']['bias'] = DataHandler.tensor_split(
        raw_data=res2_conv2_bias_raw_data, split_dict=res2_conv2_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res2_conv2_output_split_dict[(core_x, core_y)] = ((0, res2_conv2_output_raw_data.shape[0] // 2),
                                                              (0, res2_conv2_output_raw_data.shape[1]),
                                                              (0, res2_conv2_output_raw_data.shape[2]))
        else:
            res2_conv2_output_split_dict[(core_x, core_y)] = ((res2_conv2_output_raw_data.shape[0] // 2, res2_conv2_output_raw_data.shape[0]),
                                                              (0, res2_conv2_output_raw_data.shape[1]),
                                                              (0, res2_conv2_output_raw_data.shape[2]))
    data['res2']['conv2']['output'] = DataHandler.tensor_split(
        raw_data=res2_conv2_output_raw_data, split_dict=res2_conv2_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res2_add_output_raw_data = np.array(handler.parameters['res2']['add_cut']['output']).astype(np.int8)
    res2_add_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        res2_add_output_split_dict[(core_x, core_y)] = ((0, res2_add_output_raw_data.shape[0]),
                                                        (0, res2_add_output_raw_data.shape[1]),
                                                        (0, res2_add_output_raw_data.shape[2]))
    data['res2']['add']['output'] = DataHandler.tensor_split(
        raw_data=res2_add_output_raw_data, split_dict=res2_add_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])


    res3_conv1_input_raw_data = np.array(handler.parameters['res3']['conv1']['input']).astype(np.int8)
    res3_conv1_weight_raw_data = np.array(handler.parameters['res3']['conv1']['weight']).astype(np.int8)
    res3_conv1_bias_raw_data = np.array(handler.parameters['res3']['conv1']['bias']).astype(np.int32)
    res3_conv1_output_raw_data = np.array(handler.parameters['res3']['conv1_cut']['output']).astype(np.int8)
    res3_conv1_input_split_dict = {}
    res3_conv1_weight_split_dict = {}
    res3_conv1_bias_split_dict = {}
    res3_conv1_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        res3_conv1_input_split_dict[(core_x, core_y)] = ((0, res3_conv1_input_raw_data.shape[0]),
                                                         (0, res3_conv1_input_raw_data.shape[1]),
                                                         (0, res3_conv1_input_raw_data.shape[2]))
    data['res3']['conv1']['input'] = DataHandler.tensor_split(
        raw_data=res3_conv1_input_raw_data, split_dict=res3_conv1_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res3_conv1_weight_split_dict[(core_x, core_y)] = ((0, res3_conv1_weight_raw_data.shape[0]), 
                                                          (0, res3_conv1_weight_raw_data.shape[1]), 
                                                          (0, res3_conv1_weight_raw_data.shape[2]),
                                                          (0, res3_conv1_weight_raw_data.shape[3]))
    data['res3']['conv1']['weight'] = DataHandler.tensor_split(
        raw_data=res3_conv1_weight_raw_data, split_dict=res3_conv1_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        res3_conv1_bias_split_dict[(core_x, core_y)] = ((0, res3_conv1_bias_raw_data.shape[0]), )
    data['res3']['conv1']['bias'] = DataHandler.tensor_split(
        raw_data=res3_conv1_bias_raw_data, split_dict=res3_conv1_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        res3_conv1_output_split_dict[(core_x, core_y)] = ((0, res3_conv1_output_raw_data.shape[0]),
                                                          (0, res3_conv1_output_raw_data.shape[1]),
                                                          (0, res3_conv1_output_raw_data.shape[2]))
    data['res3']['conv1']['output'] = DataHandler.tensor_split(
        raw_data=res3_conv1_output_raw_data, split_dict=res3_conv1_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res3_conv2_weight_raw_data = np.array(handler.parameters['res3']['conv2']['weight']).astype(np.int8)
    res3_conv2_bias_raw_data = np.array(handler.parameters['res3']['conv2']['bias']).astype(np.int32)
    res3_conv2_output_raw_data = np.array(handler.parameters['res3']['conv2_cut']['output']).astype(np.int8)
    res3_conv2_weight_split_dict = {}
    res3_conv2_bias_split_dict = {}
    res3_conv2_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res3_conv2_weight_split_dict[(core_x, core_y)] = ((0, res3_conv2_weight_raw_data.shape[0] // 2), 
                                                              (0, res3_conv2_weight_raw_data.shape[1]), 
                                                              (0, res3_conv2_weight_raw_data.shape[2]),
                                                              (0, res3_conv2_weight_raw_data.shape[3]))
        else:
            res3_conv2_weight_split_dict[(core_x, core_y)] = ((res3_conv2_weight_raw_data.shape[0] // 2, res3_conv2_weight_raw_data.shape[0]), 
                                                              (0, res3_conv2_weight_raw_data.shape[1]), 
                                                              (0, res3_conv2_weight_raw_data.shape[2]),
                                                              (0, res3_conv2_weight_raw_data.shape[3]))
    data['res3']['conv2']['weight'] = DataHandler.tensor_split(
        raw_data=res3_conv2_weight_raw_data, split_dict=res3_conv2_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res3_conv2_bias_split_dict[(core_x, core_y)] = ((0, res3_conv2_bias_raw_data.shape[0] // 2), )
        else:
            res3_conv2_bias_split_dict[(core_x, core_y)] = ((res3_conv2_bias_raw_data.shape[0] // 2, res3_conv2_bias_raw_data.shape[0]), )
    data['res3']['conv2']['bias'] = DataHandler.tensor_split(
        raw_data=res3_conv2_bias_raw_data, split_dict=res3_conv2_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 0:
            res3_conv2_output_split_dict[(core_x, core_y)] = ((0, res3_conv2_output_raw_data.shape[0] // 2),
                                                              (0, res3_conv2_output_raw_data.shape[1]),
                                                              (0, res3_conv2_output_raw_data.shape[2]))
        else:
            res3_conv2_output_split_dict[(core_x, core_y)] = ((res3_conv2_output_raw_data.shape[0] // 2, res3_conv2_output_raw_data.shape[0]),
                                                              (0, res3_conv2_output_raw_data.shape[1]),
                                                              (0, res3_conv2_output_raw_data.shape[2]))
    data['res3']['conv2']['output'] = DataHandler.tensor_split(
        raw_data=res3_conv2_output_raw_data, split_dict=res3_conv2_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    res3_add_output_raw_data = np.array(handler.parameters['res3']['add_cut']['output']).astype(np.int8)
    res3_add_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        res3_add_output_split_dict[(core_x, core_y)] = ((0, res3_add_output_raw_data.shape[0]),
                                                        (0, res3_add_output_raw_data.shape[1]),
                                                        (0, res3_add_output_raw_data.shape[2]))
    data['res3']['add']['output'] = DataHandler.tensor_split(
        raw_data=res3_add_output_raw_data, split_dict=res3_add_output_split_dict, data_type=1,
        alignment=(32, None, None), dims=[0, 2, 1])

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g5_data(handler, size_y=1, size_x=2)
