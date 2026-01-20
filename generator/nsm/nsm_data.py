from itertools import product
import sys
import os
from numpy import core
from numpy.core.fromnumeric import size
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
import numpy as np
from generator.mapping_utils.data_handler import DataHandler


def generate_nsm_data(handler, size_y, size_x):
    data = {
        'linear_tt': {},
        'linear_st': {},
        'linear_ts': {},
        'linear_ss': {}
    }

    tt_weight = np.array(handler.parameters['linear_tt']['weight']).astype(np.int8)
    st_weight = np.array(handler.parameters['linear_st']['weight']).astype(np.int8)
    ts_weight = np.array(handler.parameters['linear_ts']['weight']).astype(np.int8)
    ss_weight = np.array(handler.parameters['linear_ss']['weight']).astype(np.int8)
    tt_weight_split_dict, st_weight_split_dict, ts_weight_split_dict, ss_weight_split_dict = {}, {}, {}, {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        tt_weight_split_dict[(core_x, core_y)] = ((0, tt_weight.shape[0]), (0, tt_weight.shape[1]))
        st_weight_split_dict[(core_x, core_y)] = ((0, st_weight.shape[0]), (0, st_weight.shape[1]))
        ts_weight_split_dict[(core_x, core_y)] = ((0, ts_weight.shape[0]), (0, ts_weight.shape[1]))
        ss_weight_split_dict[(core_x, core_y)] = ((0, ss_weight.shape[0]), (0, ss_weight.shape[1]))

    data['linear_tt']['weight'] = DataHandler.tensor_split(
        raw_data=tt_weight, split_dict=tt_weight_split_dict, data_type=1,
        alignment=(32, None), dims=[0, 1], is_weight=True)
    data['linear_st']['weight'] = DataHandler.tensor_split(
        raw_data=st_weight, split_dict=st_weight_split_dict, data_type=1,
        alignment=(32, None), dims=[0, 1], is_weight=True)
    data['linear_ts']['weight'] = DataHandler.tensor_split(
        raw_data=ts_weight, split_dict=ts_weight_split_dict, data_type=1,
        alignment=(32, None), dims=[0, 1], is_weight=True)
    data['linear_ss']['weight'] = DataHandler.tensor_split(
        raw_data=ss_weight, split_dict=ss_weight_split_dict, data_type=1,
        alignment=(32, None), dims=[0, 1], is_weight=True)

    init_state_raw_data = np.array(handler.parameters['init_state']).astype(np.int8)
    inputs_raw_data = np.array(handler.parameters['inputs']).astype(np.int8)
    t1_cut_raw_data = np.array(handler.parameters['t1_cut']).astype(np.int8)
    t2_cut_raw_data = np.array(handler.parameters['t2_cut']).astype(np.int8)
    hidden_1_cut_raw_data = np.array(handler.parameters['hidden_1_cut']).astype(np.int8)
    t3_raw_data = np.array(handler.parameters['t3']).astype(np.int32)
    t4_raw_data = np.array(handler.parameters['t4']).astype(np.int32)
    hidden_2_raw_data = np.array(handler.parameters['hidden_2']).astype(np.int32)
    act_fun_raw_data = np.array(handler.parameters['act_fun']).astype(np.int8)
    output_raw_data = np.array(handler.parameters['output']).astype(np.int8)

    init_state_split_dict = {}
    inputs_split_dict = {}
    t1_cut_split_dict = {}
    t2_cut_split_dict = {}
    hidden_1_cut_split_dict = {}
    t3_split_dict = {}
    t4_split_dict = {}
    hidden_2_split_dict = {}
    act_fun_split_dict = {}
    output_split_dict = {}

    for core_y, core_x in product(range(size_y), range(size_x)):
        init_state_split_dict[(core_x, core_y)] = ((0, 1), (0, init_state_raw_data.shape[1]))
        t1_cut_split_dict[(core_x, core_y)] = ((0, 1), (0, t1_cut_raw_data.shape[1]))
        t2_cut_split_dict[(core_x, core_y)] = ((0, 1), (0, t2_cut_raw_data.shape[1]))
        inputs_split_dict[(core_x, core_y)] = ((0, 1), (0, inputs_raw_data.shape[1]))
        hidden_1_cut_split_dict[(core_x, core_y)] = ((0, 1), (0, hidden_1_cut_raw_data.shape[1]))
        t3_split_dict[(core_x, core_y)] = ((0, 1), (0, t3_raw_data.shape[1]))
        t4_split_dict[(core_x, core_y)] = ((0, 1), (0, t4_raw_data.shape[1]))
        hidden_2_split_dict[(core_x, core_y)] = ((0, 1), (0, hidden_2_raw_data.shape[1]))
        act_fun_split_dict[(core_x, core_y)] = ((0, 1), (0, act_fun_raw_data.shape[1]))
        output_split_dict[(core_x, core_y)] = ((0, 1), (0, output_raw_data.shape[1]), (0, output_raw_data.shape[2]))

    data['init_state'] = DataHandler.tensor_split(raw_data=init_state_raw_data, split_dict=init_state_split_dict,
                                                  data_type=1,
                                                  alignment=(None, 16), dims=[1, 0])
    data['inputs'] = DataHandler.tensor_split(
        raw_data=inputs_raw_data, split_dict=inputs_split_dict,
        data_type=1, alignment=(None, 16), dims=[1, 0])
    data['t1_cut'] = DataHandler.tensor_split(
        raw_data=t1_cut_raw_data, split_dict=t1_cut_split_dict,
        data_type=1, alignment=(None, 32), dims=[1, 0])
    data['t2_cut'] = DataHandler.tensor_split(
        raw_data=t2_cut_raw_data, split_dict=t2_cut_split_dict,
        data_type=1, alignment=(None, 32), dims=[1, 0])
    data['hidden_1_cut'] = DataHandler.tensor_split(
        raw_data=hidden_1_cut_raw_data, split_dict=hidden_1_cut_split_dict,
        data_type=1, alignment=(None, 16), dims=[1, 0])
    data['t3'] = DataHandler.tensor_split(
        raw_data=t3_raw_data, split_dict=t3_split_dict,
        data_type=0, alignment=(None, 32), dims=[1, 0])
    data['t4'] = DataHandler.tensor_split(
        raw_data=t4_raw_data, split_dict=t4_split_dict,
        data_type=0, alignment=(None, 32), dims=[1, 0])
    data['hidden_2'] = DataHandler.tensor_split(
        raw_data=hidden_2_raw_data, split_dict=hidden_2_split_dict,
        data_type=0, alignment=(None, 16), dims=[1, 0])
    data['act_fun'] = DataHandler.tensor_split(
        raw_data=act_fun_raw_data, split_dict=act_fun_split_dict,
        data_type=1, alignment=(None, 16), dims=[1, 0])
    data['output'] = DataHandler.tensor_split(
        raw_data=output_raw_data, split_dict=output_split_dict,
        data_type=1, alignment=(None, None, 16), dims=[2, 1, 0])

    return data
