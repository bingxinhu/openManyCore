from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g6_data(handler, size_y, size_x):
    data = {}
    data['conv4'] = {}
    data['avgpool'] = {}

    
    conv4_input_raw_data = np.array(handler.parameters['conv4']['input']).astype(np.int8)
    conv4_weight_raw_data = np.array(handler.parameters['conv4']['weight']).astype(np.int8)
    conv4_bias_raw_data = np.array(handler.parameters['conv4']['bias']).astype(np.int32)
    conv4_output_raw_data = np.array(handler.parameters['conv4']['output']).astype(np.int32)
    avgpool_output_raw_data = np.array(handler.parameters['avgpool']['output']).astype(np.int32)
    conv4_input_split_dict = {}
    conv4_weight_split_dict = {}
    conv4_bias_split_dict = {}
    conv4_output_split_dict = {}
    avgpool_output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv4_input_split_dict[(core_x, core_y)] = ((0, conv4_input_raw_data.shape[0]),
                                                    (0, conv4_input_raw_data.shape[1]),
                                                    (0, conv4_input_raw_data.shape[2]))
    data['conv4']['input'] = DataHandler.tensor_split(
        raw_data=conv4_input_raw_data, split_dict=conv4_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv4_weight_split_dict[(core_x, core_y)] = ((0, conv4_weight_raw_data.shape[0]), 
                                                     (0, conv4_weight_raw_data.shape[1]), 
                                                     (0, conv4_weight_raw_data.shape[2]),
                                                     (0, conv4_weight_raw_data.shape[3]))
    data['conv4']['weight'] = DataHandler.tensor_split(
        raw_data=conv4_weight_raw_data, split_dict=conv4_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=[0, 1, 3, 2], is_weight=True)

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv4_bias_split_dict[(core_x, core_y)] = ((0, conv4_bias_raw_data.shape[0]), )
    data['conv4']['bias'] = DataHandler.tensor_split(
        raw_data=conv4_bias_raw_data, split_dict=conv4_bias_split_dict, data_type=0,
        alignment=(32, ), dims=[0])

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv4_output_split_dict[(core_x, core_y)] = ((0, conv4_output_raw_data.shape[0]),
                                                     (0, conv4_output_raw_data.shape[1]),
                                                     (0, conv4_output_raw_data.shape[2]))
    data['conv4']['output'] = DataHandler.tensor_split(
        raw_data=conv4_output_raw_data, split_dict=conv4_output_split_dict, data_type=0,
        alignment=(32, None, None), dims=[0, 2, 1])

    for core_y, core_x in product(range(size_y), range(size_x)):
        avgpool_output_split_dict[(core_x, core_y)] = ((0, avgpool_output_raw_data.shape[0]),
                                                       (0, avgpool_output_raw_data.shape[1]),
                                                       (0, avgpool_output_raw_data.shape[2]))
    data['avgpool']['output'] = DataHandler.tensor_split(
        raw_data=avgpool_output_raw_data, split_dict=avgpool_output_split_dict, data_type=0,
        alignment=(32, None, None), dims=[0, 2, 1])     # TEMP for sdNet

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g6_data(handler, size_y=1, size_x=1)
