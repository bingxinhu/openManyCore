from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.detection.detection_data_handler import DetectionDataHandler
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_g3_data(handler, size_y, size_x):
    data = {
        'conv2': {},
        'maxpool2': {}
    }

    # input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    conv2_input_raw_data = np.array(handler.parameters['conv2']['input']).astype(np.int8)
    conv2_weight_raw_data = np.array(handler.parameters['conv2']['weight']).astype(np.int8)
    conv2_bias_raw_data = np.array(handler.parameters['conv2']['bias']).astype(np.int32)
    maxpool2_output_raw_data = np.array(handler.parameters['maxpool2']['output']).astype(np.int8)
    conv2_input_split_dict = {}
    conv2_weight_split_dict = {}
    conv2_bias_split_dict1, conv2_bias_split_dict2 = {}, {}
    maxpool2_output_split_dict = {}
    maxpool2_output_split_dict1 = {}
    maxpool2_output_split_dict2 = {}

    conv2_input1_split_dict = {}
    conv2_input2_split_dict = {}
    conv2_input3_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        conv2_input1_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                     (core_x * size_x, core_x * size_x + 3),
                                                     (0, conv2_input_raw_data.shape[2]))
        conv2_input2_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                     (core_x * size_x + 3, core_x * size_x + 6),
                                                     (0, conv2_input_raw_data.shape[2]))
        if core_x == size_x - 1:
            conv2_input3_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                         (core_x * size_x + 6, core_x * size_x + 7),
                                                         (0, conv2_input_raw_data.shape[2]))
        else:
            conv2_input3_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                         (core_x * size_x + 6, core_x * size_x + 8),
                                                         (0, conv2_input_raw_data.shape[2]))                         
    data['conv2']['input1'] = DataHandler.tensor_split(
        raw_data=conv2_input_raw_data, split_dict=conv2_input1_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['conv2']['input2'] = DataHandler.tensor_split(
        raw_data=conv2_input_raw_data, split_dict=conv2_input2_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['conv2']['input3'] = DataHandler.tensor_split(
        raw_data=conv2_input_raw_data, split_dict=conv2_input3_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == 7:
            conv2_input_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                        (core_x * 8, (core_x + 1) * 8 - 1),
                                                        (0, conv2_input_raw_data.shape[2]))
        else:
            conv2_input_split_dict[(core_x, core_y)] = ((0, conv2_input_raw_data.shape[0]),
                                                        (core_x * 8, (core_x + 1) * 8),
                                                        (0, conv2_input_raw_data.shape[2]))
        conv2_weight_split_dict[(core_x, core_y)] = (((core_x % 2) * 64, ((core_x % 2) + 1) * 64),
                                                     (0, conv2_weight_raw_data.shape[1]),
                                                     (0, conv2_weight_raw_data.shape[2]),
                                                     (0, conv2_weight_raw_data.shape[3]))
        conv2_bias_split_dict1[(core_x, core_y)] = (((core_x % 2) * 64, (core_x % 2 + 1) * 64),)
        conv2_bias_split_dict2[(core_x, core_y)] = (((1 - core_x % 2) * 64, (2 - core_x % 2) * 64),)
        maxpool2_output_split_dict[(core_x, core_y)] = ((0, maxpool2_output_raw_data.shape[0]),
                                                        (core_x * 2, (core_x + 1) * 2 - (core_x // 7)),
                                                        (0, maxpool2_output_raw_data.shape[2]))
        maxpool2_output_split_dict1[(core_x, core_y)] = (((core_x % 2) * 64, (core_x % 2 + 1) * 64),
                                                         (core_x * 2, (core_x + 1) * 2 - (core_x // 7)),
                                                         (0, maxpool2_output_raw_data.shape[2]))
        maxpool2_output_split_dict2[(core_x, core_y)] = (((1 - core_x % 2) * 64, (2 - core_x % 2) * 64),
                                                         (core_x * 2, (core_x + 1) * 2 - (core_x // 7)),
                                                         (0, maxpool2_output_raw_data.shape[2]))
    data['conv2']['input'] = DataHandler.tensor_split(
        raw_data=conv2_input_raw_data, split_dict=conv2_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['conv2']['weight'] = DataHandler.tensor_split(
        raw_data=conv2_weight_raw_data, split_dict=conv2_weight_split_dict, data_type=1,
        alignment=(32, None, None, None), dims=(0, 1, 3, 2), is_weight=True)
    data['conv2']['bias1'] = DataHandler.tensor_split(
        raw_data=conv2_bias_raw_data, split_dict=conv2_bias_split_dict1, data_type=0,
        alignment=(32,), dims=[0])
    data['conv2']['bias2'] = DataHandler.tensor_split(
        raw_data=conv2_bias_raw_data, split_dict=conv2_bias_split_dict2, data_type=0,
        alignment=(32,), dims=[0])
    data['maxpool2']['output'] = DataHandler.tensor_split(
        raw_data=maxpool2_output_raw_data, split_dict=maxpool2_output_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['maxpool2']['output1'] = DataHandler.tensor_split(
        raw_data=maxpool2_output_raw_data, split_dict=maxpool2_output_split_dict1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['maxpool2']['output2'] = DataHandler.tensor_split(
        raw_data=maxpool2_output_raw_data, split_dict=maxpool2_output_split_dict2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g3_data(handler, size_y=1, size_x=8)
    yy = 1
