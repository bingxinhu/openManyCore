import warnings
from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.sound_tracking.utils import get_core_id
import numpy as np
import copy


def generate_gru_data(handler, size_y, size_x):
    gru_data = {}
    gru_data['forward'] = {}
    gru_data['backward'] = {}
    gru_data['gru_router'] = {}
    gru_data['gru_init'] = {}
    gru_data['pad_127_for_input'] = [[127, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                     [127, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    gru_data['gru_router']['data_in'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # 用于交互的数据
            router_raw_data_raw = np.array(handler.parameters()['maxpool']['output'].permute(1, 2, 0).reshape(-1),
                                           np.int8).reshape((39, 16))
            router_raw_data = np.zeros((39 * 2, 16), dtype=np.int8)
            router_raw_data[::2, :] = router_raw_data_raw
            router_raw_data[1::2, 0] = 127
            router_raw_data = router_raw_data.flatten()
            router_split_dict = {}
            router_split_dict[(core_x, core_y)] = ((0, router_raw_data.shape[0]),)

            gru_data['gru_router']['data_in'] = {
                **gru_data['gru_router']['data_in'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=router_raw_data,
                    split_dict=router_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['gru_router']['data_ciso'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # 用于交互的数据
            pad = torch.zeros((16,))
            pad[0] = 127
            torch_data = torch.cat((handler.parameters()['gru'][idx]['forward']['next_hid_cut']['output'],
                                    pad,
                                    handler.parameters()['gru'][idx]['backward']['next_hid_cut']['output'],
                                    pad)
                                   )
            router_raw_data = np.array(torch_data, np.int8)
            router_split_dict = {}
            router_split_dict[(core_x, core_y)] = ((0, router_raw_data.shape[0]),)

            gru_data['gru_router']['data_ciso'] = {
                **gru_data['gru_router']['data_ciso'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=router_raw_data,
                    split_dict=router_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['gru_init']['x'] = copy.deepcopy(gru_data['gru_router']['data_in'])

    gru_data['gru_init']['h'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # 用于交互的数据
            if idx == 0:
                torch_data = torch.zeros(handler.gru_hidden_size * 2 + 32)
            else:
                pad = torch.zeros((16,))
                pad[0] = 127
                torch_data = torch.cat((handler.parameters()['gru'][idx - 1]['forward']['next_hid_cut']['output'],
                                        pad,
                                        handler.parameters()['gru'][idx - 1]['backward']['next_hid_cut']['output'],
                                        pad))
            init_raw_data = np.array(torch_data, np.int8)
            init_split_dict = {}
            init_split_dict[(core_x, core_y)] = ((0, init_raw_data.shape[0]),)

            gru_data['gru_init']['h'] = {
                **gru_data['gru_init']['h'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=init_raw_data,
                    split_dict=init_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU1_input'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 input
            GRU1_input_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU1']['input'], np.int8)
            GRU1_input_raw_data = np.append(GRU1_input_raw_data,
                                            np.array([127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
            GRU1_input_split_dict = {}
            GRU1_input_split_dict[(core_x, core_y)] = ((0, GRU1_input_raw_data.shape[0]),)

            gru_data['forward']['GRU1_input'] = {
                **gru_data['forward']['GRU1_input'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_input_raw_data,
                    split_dict=GRU1_input_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU1_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 weight
            GRU1_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU1']['weight'], np.int8)
            GRU1_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU1']['bias'] / 127
            if not (GRU1_bias_raw_data == GRU1_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU1_bias_raw_data = np.array(GRU1_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU1_weight_raw_data = np.concatenate((GRU1_weight_raw_data, GRU1_bias_raw_data.reshape((128, 1))), axis=1)

            GRU1_weight_split_dict = {}
            GRU1_weight_split_dict[(core_x, core_y)] = ((0, GRU1_weight_raw_data.shape[0]),
                                                        (0, GRU1_weight_raw_data.shape[1]))

            gru_data['forward']['GRU1_weight'] = {
                **gru_data['forward']['GRU1_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_weight_raw_data,
                    split_dict=GRU1_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU1_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 bias
            GRU1_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU1']['bias'], np.int32)
            GRU1_bias_split_dict = {}
            GRU1_bias_split_dict[(core_x, core_y)] = ((0, GRU1_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU1_bias'] = {
                **gru_data['forward']['GRU1_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_bias_raw_data,
                    split_dict=GRU1_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU1_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 output
            GRU1_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU1']['output'], np.int32)
            GRU1_output_split_dict = {}
            GRU1_output_split_dict[(core_x, core_y)] = ((0, GRU1_output_raw_data.shape[0]),)

            gru_data['forward']['GRU1_output'] = {
                **gru_data['forward']['GRU1_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_output_raw_data,
                    split_dict=GRU1_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU2_input'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 input
            GRU2_input_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU2']['input'], np.int8)
            GRU2_input_raw_data = np.append(GRU2_input_raw_data,
                                            np.array([127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
            GRU2_input_split_dict = {}
            GRU2_input_split_dict[(core_x, core_y)] = ((0, GRU2_input_raw_data.shape[0]),)

            gru_data['forward']['GRU2_input'] = {
                **gru_data['forward']['GRU2_input'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_input_raw_data,
                    split_dict=GRU2_input_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU2_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 weight
            GRU2_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU2']['weight'], np.int8)
            GRU2_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU2']['bias'] / 127
            if not (GRU2_bias_raw_data == GRU2_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU2_bias_raw_data = np.array(GRU2_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU2_weight_raw_data = np.concatenate((GRU2_weight_raw_data, GRU2_bias_raw_data.reshape((128, 1))), axis=1)

            GRU2_weight_split_dict = {}
            GRU2_weight_split_dict[(core_x, core_y)] = ((0, GRU2_weight_raw_data.shape[0]),
                                                        (0, GRU2_weight_raw_data.shape[1]))

            gru_data['forward']['GRU2_weight'] = {
                **gru_data['forward']['GRU2_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_weight_raw_data,
                    split_dict=GRU2_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU2_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 bias
            GRU2_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU2']['bias'], np.int32)
            GRU2_bias_split_dict = {}
            GRU2_bias_split_dict[(core_x, core_y)] = ((0, GRU2_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU2_bias'] = {
                **gru_data['forward']['GRU2_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_bias_raw_data,
                    split_dict=GRU2_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU2_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 output
            GRU2_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU2']['output'], np.int32)
            GRU2_output_split_dict = {}
            GRU2_output_split_dict[(core_x, core_y)] = ((0, GRU2_output_raw_data.shape[0]),)

            gru_data['forward']['GRU2_output'] = {
                **gru_data['forward']['GRU2_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_output_raw_data,
                    split_dict=GRU2_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['r_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # r output
            r_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['r']['output'], np.int8)
            r_output_split_dict = {}
            r_output_split_dict[(core_x, core_y)] = ((0, r_output_raw_data.shape[0]),)

            gru_data['forward']['r_output'] = {
                **gru_data['forward']['r_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=r_output_raw_data,
                    split_dict=r_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU4_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 weight
            GRU4_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU4']['weight'], np.int8)
            GRU4_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU4']['bias'] / 127
            if not (GRU4_bias_raw_data == GRU4_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU4_bias_raw_data = np.array(GRU4_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU4_weight_raw_data = np.concatenate((GRU4_weight_raw_data, GRU4_bias_raw_data.reshape((128, 1))), axis=1)

            GRU4_weight_split_dict = {}
            GRU4_weight_split_dict[(core_x, core_y)] = ((0, GRU4_weight_raw_data.shape[0]),
                                                        (0, GRU4_weight_raw_data.shape[1]))

            gru_data['forward']['GRU4_weight'] = {
                **gru_data['forward']['GRU4_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_weight_raw_data,
                    split_dict=GRU4_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU4_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 bias
            GRU4_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU4']['bias'], np.int32)
            GRU4_bias_split_dict = {}
            GRU4_bias_split_dict[(core_x, core_y)] = ((0, GRU4_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU4_bias'] = {
                **gru_data['forward']['GRU4_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_bias_raw_data,
                    split_dict=GRU4_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU4_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 output
            GRU4_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU4']['output'], np.int32)
            GRU4_output_split_dict = {}
            GRU4_output_split_dict[(core_x, core_y)] = ((0, GRU4_output_raw_data.shape[0]),)

            gru_data['forward']['GRU4_output'] = {
                **gru_data['forward']['GRU4_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_output_raw_data,
                    split_dict=GRU4_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU5_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 weight
            GRU5_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU5']['weight'], np.int8)
            GRU5_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU5']['bias'] / 127
            if not (GRU5_bias_raw_data == GRU5_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU5_bias_raw_data = np.array(GRU5_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU5_weight_raw_data = np.concatenate((GRU5_weight_raw_data, GRU5_bias_raw_data.reshape((128, 1))), axis=1)

            GRU5_weight_split_dict = {}
            GRU5_weight_split_dict[(core_x, core_y)] = ((0, GRU5_weight_raw_data.shape[0]),
                                                        (0, GRU5_weight_raw_data.shape[1]))

            gru_data['forward']['GRU5_weight'] = {
                **gru_data['forward']['GRU5_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_weight_raw_data,
                    split_dict=GRU5_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU5_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 bias
            GRU5_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU5']['bias'], np.int32)
            GRU5_bias_split_dict = {}
            GRU5_bias_split_dict[(core_x, core_y)] = ((0, GRU5_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU5_bias'] = {
                **gru_data['forward']['GRU5_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_bias_raw_data,
                    split_dict=GRU5_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU5_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 output
            GRU5_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU5']['output'], np.int32)
            GRU5_output_split_dict = {}
            GRU5_output_split_dict[(core_x, core_y)] = ((0, GRU5_output_raw_data.shape[0]),)

            gru_data['forward']['GRU5_output'] = {
                **gru_data['forward']['GRU5_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_output_raw_data,
                    split_dict=GRU5_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['zt_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # Zt output
            zt_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['z']['output'], np.int8)
            zt_output_split_dict = {}
            zt_output_split_dict[(core_x, core_y)] = ((0, zt_output_raw_data.shape[0]),)

            gru_data['forward']['zt_output'] = {
                **gru_data['forward']['zt_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=zt_output_raw_data,
                    split_dict=zt_output_split_dict,
                    data_type=1, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU7_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 weight
            GRU7_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU7']['weight'], np.int8)
            GRU7_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU7']['bias'] / 127
            if not (GRU7_bias_raw_data == GRU7_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU7_bias_raw_data = np.array(GRU7_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU7_weight_raw_data = np.concatenate((GRU7_weight_raw_data, GRU7_bias_raw_data.reshape((128, 1))), axis=1)

            GRU7_weight_split_dict = {}
            GRU7_weight_split_dict[(core_x, core_y)] = ((0, GRU7_weight_raw_data.shape[0]),
                                                        (0, GRU7_weight_raw_data.shape[1]))

            gru_data['forward']['GRU7_weight'] = {
                **gru_data['forward']['GRU7_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_weight_raw_data,
                    split_dict=GRU7_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU7_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 bias
            GRU7_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU7']['bias'], np.int32)
            GRU7_bias_split_dict = {}
            GRU7_bias_split_dict[(core_x, core_y)] = ((0, GRU7_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU7_bias'] = {
                **gru_data['forward']['GRU7_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_bias_raw_data,
                    split_dict=GRU7_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU7_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 output
            GRU7_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU7']['output'], np.int32)
            GRU7_output_split_dict = {}
            GRU7_output_split_dict[(core_x, core_y)] = ((0, GRU7_output_raw_data.shape[0]),)

            gru_data['forward']['GRU7_output'] = {
                **gru_data['forward']['GRU7_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_output_raw_data,
                    split_dict=GRU7_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU8_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 weight
            GRU8_weight_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU8']['weight'], np.int8)
            GRU8_bias_raw_data = handler.parameters()['gru'][idx]['forward']['GRU8']['bias'] / 127
            if not (GRU8_bias_raw_data == GRU8_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU8_bias_raw_data = np.array(GRU8_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU8_weight_raw_data = np.concatenate((GRU8_weight_raw_data, GRU8_bias_raw_data.reshape((128, 1))), axis=1)

            GRU8_weight_split_dict = {}
            GRU8_weight_split_dict[(core_x, core_y)] = ((0, GRU8_weight_raw_data.shape[0]),
                                                        (0, GRU8_weight_raw_data.shape[1]))

            gru_data['forward']['GRU8_weight'] = {
                **gru_data['forward']['GRU8_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_weight_raw_data,
                    split_dict=GRU8_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['forward']['GRU8_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 bias
            GRU8_bias_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU8']['bias'], np.int32)
            GRU8_bias_split_dict = {}
            GRU8_bias_split_dict[(core_x, core_y)] = ((0, GRU8_bias_raw_data.shape[0]),)

            gru_data['forward']['GRU8_bias'] = {
                **gru_data['forward']['GRU8_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_bias_raw_data,
                    split_dict=GRU8_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU8_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 output
            GRU8_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU8_cut']['output'], np.int8)
            GRU8_output_split_dict = {}
            GRU8_output_split_dict[(core_x, core_y)] = ((0, GRU8_output_raw_data.shape[0]),)

            gru_data['forward']['GRU8_output'] = {
                **gru_data['forward']['GRU8_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_output_raw_data,
                    split_dict=GRU8_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU9_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU9 output
            GRU9_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU9']['output'], np.int32)
            GRU9_output_split_dict = {}
            GRU9_output_split_dict[(core_x, core_y)] = ((0, GRU9_output_raw_data.shape[0]),)

            gru_data['forward']['GRU9_output'] = {
                **gru_data['forward']['GRU9_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU9_output_raw_data,
                    split_dict=GRU9_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['n_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # n output
            n_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['n']['output'], np.int8)
            n_output_split_dict = {}
            n_output_split_dict[(core_x, core_y)] = ((0, n_output_raw_data.shape[0]),)

            gru_data['forward']['n_output'] = {
                **gru_data['forward']['n_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=n_output_raw_data,
                    split_dict=n_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU11_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU11 output
            GRU11_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU11_cut']['output'],
                                             np.int8)
            GRU11_output_split_dict = {}
            GRU11_output_split_dict[(core_x, core_y)] = ((0, GRU11_output_raw_data.shape[0]),)

            gru_data['forward']['GRU11_output'] = {
                **gru_data['forward']['GRU11_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU11_output_raw_data,
                    split_dict=GRU11_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['forward']['GRU12_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU12 output
            GRU12_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU12']['output'], np.int32)
            GRU12_output_split_dict = {}
            GRU12_output_split_dict[(core_x, core_y)] = ((0, GRU12_output_raw_data.shape[0]),)

            gru_data['forward']['GRU12_output'] = {
                **gru_data['forward']['GRU12_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU12_output_raw_data,
                    split_dict=GRU12_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['GRU13_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU13 output
            GRU13_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['GRU13']['output'], np.int32)
            GRU13_output_split_dict = {}
            GRU13_output_split_dict[(core_x, core_y)] = ((0, GRU13_output_raw_data.shape[0]),)

            gru_data['forward']['GRU13_output'] = {
                **gru_data['forward']['GRU13_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU13_output_raw_data,
                    split_dict=GRU13_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['forward']['next_hid_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # next hid output
            next_hid_output_raw_data = np.array(handler.parameters()['gru'][idx]['forward']['next_hid_cut']['output'],
                                                np.int8)
            next_hid_output_split_dict = {}
            next_hid_output_split_dict[(core_x, core_y)] = ((0, next_hid_output_raw_data.shape[0]),)

            gru_data['forward']['next_hid_output'] = {
                **gru_data['forward']['next_hid_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=next_hid_output_raw_data,
                    split_dict=next_hid_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU1_input'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 input
            GRU1_input_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU1']['input'], np.int8)
            GRU1_input_raw_data = np.append(GRU1_input_raw_data,
                                            np.array([127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
            GRU1_input_split_dict = {}
            GRU1_input_split_dict[(core_x, core_y)] = ((0, GRU1_input_raw_data.shape[0]),)

            gru_data['backward']['GRU1_input'] = {
                **gru_data['backward']['GRU1_input'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_input_raw_data,
                    split_dict=GRU1_input_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU1_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 weight
            GRU1_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU1']['weight'], np.int8)
            GRU1_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU1']['bias'] / 127
            if not (GRU1_bias_raw_data == GRU1_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU1_bias_raw_data = np.array(GRU1_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU1_weight_raw_data = np.concatenate((GRU1_weight_raw_data, GRU1_bias_raw_data.reshape((128, 1))), axis=1)

            GRU1_weight_split_dict = {}
            GRU1_weight_split_dict[(core_x, core_y)] = ((0, GRU1_weight_raw_data.shape[0]),
                                                        (0, GRU1_weight_raw_data.shape[1]))

            gru_data['backward']['GRU1_weight'] = {
                **gru_data['backward']['GRU1_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_weight_raw_data,
                    split_dict=GRU1_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU1_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 bias
            GRU1_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU1']['bias'], np.int32)
            GRU1_bias_split_dict = {}
            GRU1_bias_split_dict[(core_x, core_y)] = ((0, GRU1_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU1_bias'] = {
                **gru_data['backward']['GRU1_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_bias_raw_data,
                    split_dict=GRU1_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU1_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU1 output
            GRU1_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU1']['output'], np.int32)
            GRU1_output_split_dict = {}
            GRU1_output_split_dict[(core_x, core_y)] = ((0, GRU1_output_raw_data.shape[0]),)

            gru_data['backward']['GRU1_output'] = {
                **gru_data['backward']['GRU1_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU1_output_raw_data,
                    split_dict=GRU1_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU2_input'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 input
            GRU2_input_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU2']['input'], np.int8)
            GRU2_input_raw_data = np.append(GRU2_input_raw_data,
                                            np.array([127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
            GRU2_input_split_dict = {}
            GRU2_input_split_dict[(core_x, core_y)] = ((0, GRU2_input_raw_data.shape[0]),)

            gru_data['backward']['GRU2_input'] = {
                **gru_data['backward']['GRU2_input'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_input_raw_data,
                    split_dict=GRU2_input_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU2_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 weight
            GRU2_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU2']['weight'], np.int8)
            GRU2_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU2']['bias'] / 127
            if not (GRU2_bias_raw_data == GRU2_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU2_bias_raw_data = np.array(GRU2_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU2_weight_raw_data = np.concatenate((GRU2_weight_raw_data, GRU2_bias_raw_data.reshape((128, 1))), axis=1)

            GRU2_weight_split_dict = {}
            GRU2_weight_split_dict[(core_x, core_y)] = ((0, GRU2_weight_raw_data.shape[0]),
                                                        (0, GRU2_weight_raw_data.shape[1]))

            gru_data['backward']['GRU2_weight'] = {
                **gru_data['backward']['GRU2_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_weight_raw_data,
                    split_dict=GRU2_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU2_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 bias
            GRU2_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU2']['bias'], np.int32)
            GRU2_bias_split_dict = {}
            GRU2_bias_split_dict[(core_x, core_y)] = ((0, GRU2_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU2_bias'] = {
                **gru_data['backward']['GRU2_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_bias_raw_data,
                    split_dict=GRU2_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU2_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU2 output
            GRU2_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU2']['output'], np.int32)
            GRU2_output_split_dict = {}
            GRU2_output_split_dict[(core_x, core_y)] = ((0, GRU2_output_raw_data.shape[0]),)

            gru_data['backward']['GRU2_output'] = {
                **gru_data['backward']['GRU2_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU2_output_raw_data,
                    split_dict=GRU2_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['r_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # r output
            r_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['r']['output'], np.int8)
            r_output_split_dict = {}
            r_output_split_dict[(core_x, core_y)] = ((0, r_output_raw_data.shape[0]),)

            gru_data['backward']['r_output'] = {
                **gru_data['backward']['r_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=r_output_raw_data,
                    split_dict=r_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU4_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 weight
            GRU4_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU4']['weight'], np.int8)
            GRU4_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU4']['bias'] / 127
            if not (GRU4_bias_raw_data == GRU4_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU4_bias_raw_data = np.array(GRU4_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU4_weight_raw_data = np.concatenate((GRU4_weight_raw_data, GRU4_bias_raw_data.reshape((128, 1))), axis=1)

            GRU4_weight_split_dict = {}
            GRU4_weight_split_dict[(core_x, core_y)] = ((0, GRU4_weight_raw_data.shape[0]),
                                                        (0, GRU4_weight_raw_data.shape[1]))

            gru_data['backward']['GRU4_weight'] = {
                **gru_data['backward']['GRU4_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_weight_raw_data,
                    split_dict=GRU4_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU4_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 bias
            GRU4_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU4']['bias'], np.int32)
            GRU4_bias_split_dict = {}
            GRU4_bias_split_dict[(core_x, core_y)] = ((0, GRU4_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU4_bias'] = {
                **gru_data['backward']['GRU4_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_bias_raw_data,
                    split_dict=GRU4_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU4_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU4 output
            GRU4_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU4']['output'], np.int32)
            GRU4_output_split_dict = {}
            GRU4_output_split_dict[(core_x, core_y)] = ((0, GRU4_output_raw_data.shape[0]),)

            gru_data['backward']['GRU4_output'] = {
                **gru_data['backward']['GRU4_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU4_output_raw_data,
                    split_dict=GRU4_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU5_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 weight
            GRU5_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU5']['weight'], np.int8)
            GRU5_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU5']['bias'] / 127
            if not (GRU5_bias_raw_data == GRU5_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU5_bias_raw_data = np.array(GRU5_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU5_weight_raw_data = np.concatenate((GRU5_weight_raw_data, GRU5_bias_raw_data.reshape((128, 1))), axis=1)

            GRU5_weight_split_dict = {}
            GRU5_weight_split_dict[(core_x, core_y)] = ((0, GRU5_weight_raw_data.shape[0]),
                                                        (0, GRU5_weight_raw_data.shape[1]))

            gru_data['backward']['GRU5_weight'] = {
                **gru_data['backward']['GRU5_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_weight_raw_data,
                    split_dict=GRU5_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU5_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 bias
            GRU5_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU5']['bias'], np.int32)
            GRU5_bias_split_dict = {}
            GRU5_bias_split_dict[(core_x, core_y)] = ((0, GRU5_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU5_bias'] = {
                **gru_data['backward']['GRU5_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_bias_raw_data,
                    split_dict=GRU5_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU5_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU5 output
            GRU5_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU5']['output'], np.int32)
            GRU5_output_split_dict = {}
            GRU5_output_split_dict[(core_x, core_y)] = ((0, GRU5_output_raw_data.shape[0]),)

            gru_data['backward']['GRU5_output'] = {
                **gru_data['backward']['GRU5_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU5_output_raw_data,
                    split_dict=GRU5_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['zt_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # Zt output
            zt_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['z']['output'], np.int8)
            zt_output_split_dict = {}
            zt_output_split_dict[(core_x, core_y)] = ((0, zt_output_raw_data.shape[0]),)

            gru_data['backward']['zt_output'] = {
                **gru_data['backward']['zt_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=zt_output_raw_data,
                    split_dict=zt_output_split_dict,
                    data_type=1, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU7_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 weight
            GRU7_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU7']['weight'], np.int8)
            GRU7_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU7']['bias'] / 127
            if not (GRU7_bias_raw_data == GRU7_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU7_bias_raw_data = np.array(GRU7_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU7_weight_raw_data = np.concatenate((GRU7_weight_raw_data, GRU7_bias_raw_data.reshape((128, 1))), axis=1)

            GRU7_weight_split_dict = {}
            GRU7_weight_split_dict[(core_x, core_y)] = ((0, GRU7_weight_raw_data.shape[0]),
                                                        (0, GRU7_weight_raw_data.shape[1]))

            gru_data['backward']['GRU7_weight'] = {
                **gru_data['backward']['GRU7_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_weight_raw_data,
                    split_dict=GRU7_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU7_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 bias
            GRU7_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU7']['bias'], np.int32)
            GRU7_bias_split_dict = {}
            GRU7_bias_split_dict[(core_x, core_y)] = ((0, GRU7_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU7_bias'] = {
                **gru_data['backward']['GRU7_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_bias_raw_data,
                    split_dict=GRU7_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU7_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU7 output
            GRU7_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU7']['output'], np.int32)
            GRU7_output_split_dict = {}
            GRU7_output_split_dict[(core_x, core_y)] = ((0, GRU7_output_raw_data.shape[0]),)

            gru_data['backward']['GRU7_output'] = {
                **gru_data['backward']['GRU7_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU7_output_raw_data,
                    split_dict=GRU7_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU8_weight'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 weight
            GRU8_weight_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU8']['weight'], np.int8)
            GRU8_bias_raw_data = handler.parameters()['gru'][idx]['backward']['GRU8']['bias'] / 127
            if not (GRU8_bias_raw_data == GRU8_bias_raw_data.clamp(-128, 127).floor()).all():
                warnings.warn('Bias cannot be divided by 127')
            GRU8_bias_raw_data = np.array(GRU8_bias_raw_data.clamp(-128, 127).floor(), dtype=np.int8)
            GRU8_weight_raw_data = np.concatenate((GRU8_weight_raw_data, GRU8_bias_raw_data.reshape((128, 1))), axis=1)

            GRU8_weight_split_dict = {}
            GRU8_weight_split_dict[(core_x, core_y)] = ((0, GRU8_weight_raw_data.shape[0]),
                                                        (0, GRU8_weight_raw_data.shape[1]))

            gru_data['backward']['GRU8_weight'] = {
                **gru_data['backward']['GRU8_weight'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_weight_raw_data,
                    split_dict=GRU8_weight_split_dict,
                    data_type=1, alignment=(32, None),
                    dims=[0, 1], is_weight=True)
            }

    gru_data['backward']['GRU8_bias'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 bias
            GRU8_bias_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU8']['bias'], np.int32)
            GRU8_bias_split_dict = {}
            GRU8_bias_split_dict[(core_x, core_y)] = ((0, GRU8_bias_raw_data.shape[0]),)

            gru_data['backward']['GRU8_bias'] = {
                **gru_data['backward']['GRU8_bias'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_bias_raw_data,
                    split_dict=GRU8_bias_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU8_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU8 output
            GRU8_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU8_cut']['output'], np.int8)
            GRU8_output_split_dict = {}
            GRU8_output_split_dict[(core_x, core_y)] = ((0, GRU8_output_raw_data.shape[0]),)

            gru_data['backward']['GRU8_output'] = {
                **gru_data['backward']['GRU8_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU8_output_raw_data,
                    split_dict=GRU8_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU9_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU9 output
            GRU9_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU9']['output'], np.int32)
            GRU9_output_split_dict = {}
            GRU9_output_split_dict[(core_x, core_y)] = ((0, GRU9_output_raw_data.shape[0]),)

            gru_data['backward']['GRU9_output'] = {
                **gru_data['backward']['GRU9_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU9_output_raw_data,
                    split_dict=GRU9_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['n_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # n output
            n_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['n']['output'], np.int8)
            n_output_split_dict = {}
            n_output_split_dict[(core_x, core_y)] = ((0, n_output_raw_data.shape[0]),)

            gru_data['backward']['n_output'] = {
                **gru_data['backward']['n_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=n_output_raw_data,
                    split_dict=n_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU11_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU11 output
            GRU11_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU11_cut']['output'],
                                             np.int8)
            GRU11_output_split_dict = {}
            GRU11_output_split_dict[(core_x, core_y)] = ((0, GRU11_output_raw_data.shape[0]),)

            gru_data['backward']['GRU11_output'] = {
                **gru_data['backward']['GRU11_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU11_output_raw_data,
                    split_dict=GRU11_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    gru_data['backward']['GRU12_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU12 output
            GRU12_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU12']['output'], np.int32)
            GRU12_output_split_dict = {}
            GRU12_output_split_dict[(core_x, core_y)] = ((0, GRU12_output_raw_data.shape[0]),)

            gru_data['backward']['GRU12_output'] = {
                **gru_data['backward']['GRU12_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU12_output_raw_data,
                    split_dict=GRU12_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['GRU13_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # GRU13 output
            GRU13_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['GRU13']['output'], np.int32)
            GRU13_output_split_dict = {}
            GRU13_output_split_dict[(core_x, core_y)] = ((0, GRU13_output_raw_data.shape[0]),)

            gru_data['backward']['GRU13_output'] = {
                **gru_data['backward']['GRU13_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=GRU13_output_raw_data,
                    split_dict=GRU13_output_split_dict,
                    data_type=0, alignment=(32,),
                    dims=[0])
            }

    gru_data['backward']['next_hid_output'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        idx = get_core_id(core_x, core_y)

        if idx < handler.sequence_length:
            # next hid output
            next_hid_output_raw_data = np.array(handler.parameters()['gru'][idx]['backward']['next_hid_cut']['output'],
                                                np.int8)
            next_hid_output_split_dict = {}
            next_hid_output_split_dict[(core_x, core_y)] = ((0, next_hid_output_raw_data.shape[0]),)

            gru_data['backward']['next_hid_output'] = {
                **gru_data['backward']['next_hid_output'],
                **SoundTrackingDataHandler.tensor_split(
                    raw_data=next_hid_output_raw_data,
                    split_dict=next_hid_output_split_dict,
                    data_type=1, alignment=(16,),
                    dims=[0])
            }

    return gru_data


if __name__ == '__main__':
    in_cut_start_mat = [{'forward':
                             {'GRU3_cut': 3,
                              'GRU6_cut': 3,
                              'GRU8_cut': 3,
                              'GRU10_cut': 3,
                              'GRU11_cut': 3,
                              'next_hid_cut': 3},
                         'backward':
                             {'GRU3_cut': 3,
                              'GRU6_cut': 3,
                              'GRU8_cut': 3,
                              'GRU10_cut': 3,
                              'GRU11_cut': 3,
                              'next_hid_cut': 3}}
                        ] * 37
    q_one_list = [{'forward': 127, 'backward': 127}] * 37
    lut_mat = [{'forward':
                    {'sigmoid_r_d': 128,
                     'sigmoid_r_m': 128,
                     'sigmoid_z_d': 128,
                     'sigmoid_z_m': 128,
                     'tanh_d': 128,
                     'tanh_m': 128},
                'backward':
                    {'sigmoid_r_d': 128,
                     'sigmoid_r_m': 128,
                     'sigmoid_z_d': 128,
                     'sigmoid_z_m': 128,
                     'tanh_d': 128,
                     'tanh_m': 128}}
               ] * 37

    handler = SoundTrackingDataHandler(in_cut_start_mat, q_one_list, lut_mat,
                                       input_size=16, hidden_size=128, sequence_length=37)
    gru_data = generate_gru_data(handler, size_y=3, size_x=16)
