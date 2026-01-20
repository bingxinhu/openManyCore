import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p09, p06, p26, p41
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.sound_tracking_dynamic.gru_data import generate_gru_data
from generator.sound_tracking.utils import get_core_id


def gen_gru_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_mat=None, data=None, handler=None,
                       q_ones=None, lut_mat=None, delay_l4=None, delay_l5=None,
                       in_data_en=False, out_data_en=False, chip=(0, 0), init_data=None):
    """
        Sound Tracking: Bidirectional GRU
        core_x * core_y: 16 * 3 (只用到了前39个)
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        if get_core_id(core_x, core_y) < handler.sequence_length:
            map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
                'prims': []
            }
    phase_group = map_config[(chip, 0)][0]

    # ******** 数据交互 ********
    offset = 1
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_in = data['gru_router']['data_in'][(core_x, core_y)]
                data_ciso = data['gru_router']['data_ciso'][(core_x, core_y)]
                if core_x == 0 and core_y == 0:
                    axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                            addr_inb=0x1f000 >> 2, addr_bias=0x1f000 >> 2, addr_out=0x1f000 >> 2, axon_delay=True,
                            L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
                else:
                    axon = None
                # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0xff00 >> 2, length_in=39 * 16,
                            num_in=1, length_ciso=256, num_ciso=1, length_out=880, num_out=1, type_in=1, type_out=1,
                            data_in=data_in if init_data else None,
                            data_ciso=data_ciso if init_data else None)
                soma1 = None if not out_data_en else soma1
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=110 - 1,
                             receive_num=3 if core_id == 0 else 0, addr_din_base=0x1000 >> 2,
                             addr_din_length=78 - 1 if core_id == 0 else 110 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             cxy=0, nx=0, ny=0, relay_num=0, data_in=None)
                if 0 <= core_id < 15 or core_id >= 32:
                    X = 1
                    Y = 0
                elif core_id == 15 or core_id == 31:
                    X = 0
                    Y = 1
                else:
                    X = -1
                    Y = 0
                router.addRHead(S=0, T=1, P=0, Q=0, X=X, Y=Y, A=0, pack_per_Rhead=110 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p26(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x1ff00 >> 2,
                            length_in=39 * 16 if core_id == 0 else 880,
                            num_in=1, length_ciso=256, num_ciso=1, length_out=39 * 16, num_out=1, type_in=1,
                            type_out=1)
                soma2 = None if not in_data_en else soma2
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # ******** 开始计算 ********
    # ******** 正向计算 ********
    # GRU1 = Wir · xt + bir
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_x = data['forward']['GRU1_input'][(core_x, core_y)] if not in_data_en else None
                data_w = data['forward']['GRU1_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU1_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0000 + core_id * 0x0010) >> 2, addr_inb=0x10000 >> 2,
                           addr_bias=0x10800 >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2, bias_length=128,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU2 = Whr · ht-1 + bhr
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_x = data['forward']['GRU2_input'][(core_x, core_y)] if not in_data_en else None
                data_w = data['forward']['GRU2_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU2_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff00 >> 2, addr_inb=0x10a00 >> 2, addr_bias=0x14a00 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU3 = GRU1 + GRU2
    # Sigmoid(GRU3)
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_lut = handler.lut_gen('sigmoid', 1, 8, lut_mat[core_id]['forward']['sigmoid_r_d'], 8,
                                           lut_mat[core_id]['forward']['sigmoid_r_m'])
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0880 >> 2, addr_lut=0x14c00 >> 2, group_num=1,
                            neuron_num=128,
                            lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['GRU3_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=data_lut)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU4 = Wiz · xt + biz
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['forward']['GRU4_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU4_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0000 + core_id * 0x0010) >> 2, addr_inb=0x14d00 >> 2,
                           addr_bias=0x15500 >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU5 = Whz · ht-1 + bhz
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['forward']['GRU5_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU5_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff00 >> 2, addr_inb=0x15700 >> 2, addr_bias=0x19700 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Zt = Sigmoid(GRU4 + GRU5)
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0900 >> 2, addr_lut=0x14c00 >> 2, group_num=1,
                            neuron_num=128,
                            lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['GRU6_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=None)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU7 = Win · xt + bin
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['forward']['GRU7_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU7_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0000 + core_id * 0x0010) >> 2, addr_inb=0x19900 >> 2,
                           addr_bias=0x1a100 >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU8 = Cut(Whn ht-1 + bhn)
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['forward']['GRU8_weight'][(core_x, core_y)]
                data_b = data['forward']['GRU8_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff00 >> 2, addr_inb=0x1a300 >> 2, addr_bias=0x1e300 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = pX5(mode='max', addr_in=0x0480 >> 2, addr_out=0x0980 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['GRU8_cut'],
                            row_ck_on=0, in_row_max=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU9 = Rt · GRU8
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0880 >> 2, addr_inb=0x0980 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Nt = Tanh(GRU7 + GRU9)
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_lut = handler.lut_gen('tanh', 1, 8, lut_mat[core_id]['forward']['tanh_d'], 8,
                                           lut_mat[core_id]['forward']['tanh_m'])
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0880 >> 2, addr_lut=0x1e500 >> 2, group_num=1,
                            neuron_num=128, lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['GRU10_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=data_lut)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU11 = Cut(1 - Zt)
    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p83(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0900 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0280 >> 2, constant_a=-1,
                           constant_b=q_ones[core_id]['forward'], ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_b=None)
                soma1 = pX5(mode='max', addr_in=0x0280 >> 2, addr_out=0x0980 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['GRU11_cut'],
                            row_ck_on=1, in_row_max=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU12 = GRU11 * nt
    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0880 >> 2, addr_inb=0x0980 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU13 = Zt * ht-1
    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0900 >> 2, addr_inb=0x1ff00 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # ht = Cut(GRU12 + GRU13)
    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = pX5(mode='max', addr_in=0x0680 >> 2, addr_out=0xff00 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['forward']['next_hid_cut'],
                            row_ck_on=1, in_row_max=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # ******** 反向计算 ********
    # GRU1 = Wir · xt + bir
    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_x = data['backward']['GRU1_input'][(core_x, core_y)] if not in_data_en else None
                data_w = data['backward']['GRU1_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU1_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0260 - core_id * 0x0010) >> 2,
                           addr_inb=(0x10000 - 0xf000) >> 2, addr_bias=(0x10800 - 0xf000) >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2, bias_length=128,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU2 = Whr · ht-1 + bhr
    if phase_en[offset + 15]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_x = data['backward']['GRU2_input'][(core_x, core_y)] if not in_data_en else None
                data_w = data['backward']['GRU2_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU2_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff80 >> 2, addr_inb=(0x10a00 - 0xf000) >> 2,
                           addr_bias=(0x14a00 - 0xf000) >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=data_x, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU3 = GRU1 + GRU2
    # Sigmoid(GRU3)
    if phase_en[offset + 16]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_lut = handler.lut_gen('sigmoid', 1, 8, lut_mat[core_id]['backward']['sigmoid_r_d'], 8,
                                           lut_mat[core_id]['backward']['sigmoid_r_m'])
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0880 >> 2, addr_lut=(0x14c00 - 0xf000) >> 2,
                            group_num=1, neuron_num=128,
                            lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['GRU3_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=data_lut)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU4 = Wiz · xt + biz
    if phase_en[offset + 17]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['backward']['GRU4_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU4_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0260 - core_id * 0x0010) >> 2,
                           addr_inb=(0x14d00 - 0xf000) >> 2, addr_bias=(0x15500 - 0xf000) >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU5 = Whz · ht-1 + bhz
    if phase_en[offset + 18]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['backward']['GRU5_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU5_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff80 >> 2, addr_inb=(0x15700 - 0xf000) >> 2,
                           addr_bias=(0x19700 - 0xf000) >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Zt = Sigmoid(GRU4 + GRU5)
    if phase_en[offset + 19]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0900 >> 2, addr_lut=(0x14c00 - 0xf000) >> 2, group_num=1,
                            neuron_num=128,
                            lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['GRU6_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=None)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU7 = Win · xt + bin
    if phase_en[offset + 20]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['backward']['GRU7_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU7_bias'][(core_x, core_y)]
                axon = p04(cin=16, cout=128, addr_ina=(0x0260 - core_id * 0x0010) >> 2,
                           addr_inb=(0x19900 - 0xf000) >> 2, addr_bias=(0x1a100 - 0xf000) >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU8 = Cut(Whn ht-1 + bhn)
    if phase_en[offset + 21]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_w = data['backward']['GRU8_weight'][(core_x, core_y)]
                data_b = data['backward']['GRU8_bias'][(core_x, core_y)]
                axon = p04(cin=128, cout=128, addr_ina=0x1ff80 >> 2, addr_inb=(0x1a300 - 0xf000) >> 2,
                           addr_bias=(0x1e300 - 0xf000) >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, inb_type=1, load_bias=2,
                           data_x=None, data_w=data_w, data_b=data_b)
                axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                           addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                           load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None, data_w=data_w, data_b=data_b)
                soma1 = None
                router = None
                soma2 = pX5(mode='max', addr_in=0x0480 >> 2, addr_out=0x0980 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['GRU8_cut'],
                            row_ck_on=0, in_row_max=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU9 = Rt · GRU8
    if phase_en[offset + 22]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0880 >> 2, addr_inb=0x0980 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Nt = Tanh(GRU7 + GRU9)
    if phase_en[offset + 23]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                data_lut = handler.lut_gen('tanh', 1, 8, lut_mat[core_id]['backward']['tanh_d'], 8,
                                           lut_mat[core_id]['backward']['tanh_m'])
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = p07(addr_in=0x0680 >> 2, addr_out=0x0880 >> 2, addr_lut=(0x1e500 - 0xf000) >> 2, group_num=1,
                            neuron_num=128, lut_dw=1, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['GRU10_cut'],
                            row_ck_on=1, in_row_max=1, data_in=None, data_lut=data_lut)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU11 = Cut(1 - Zt)
    if phase_en[offset + 24]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p83(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0900 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0280 >> 2, constant_a=-1,
                           constant_b=q_ones[core_id]['backward'], ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_b=None)
                soma1 = pX5(mode='max', addr_in=0x0280 >> 2, addr_out=0x0980 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['GRU11_cut'],
                            row_ck_on=1, in_row_max=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU12 = GRU11 * nt
    if phase_en[offset + 25]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0880 >> 2, addr_inb=0x0980 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0280 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # GRU13 = Zt * ht-1
    if phase_en[offset + 26]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                           addr_ina=0x0900 >> 2, addr_inb=0x1ff80 >> 2, addr_bias=0x0000 >> 2,
                           addr_out=0x0480 >> 2, ina_type=1, load_bias=0, bias_length=0,
                           data_x=None, data_y=None, data_b=None)
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # ht = Cut(GRU12 + GRU13)
    if phase_en[offset + 27]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = get_core_id(core_x, core_y)
            if core_id < handler.sequence_length:
                axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=1, py=1,
                           addr_in=0x0280 >> 2, addr_bias=0x0, addr_out=0x0680 >> 2, pad_top=0, pad_down=0, pad_left=0,
                           pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
                soma1 = pX5(mode='max', addr_in=0x0680 >> 2, addr_out=0xff80 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                            in_cut_start=in_cut_start_mat[core_id]['backward']['next_hid_cut'],
                            row_ck_on=1, in_row_max=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'ST_GRU'
    chip = (0, 0)
    cuts = None
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    # phase[0] = 1
    # phase[1] = 1

    phase[phase_offset + 0] = 1
    phase[phase_offset + 1] = 1
    phase[phase_offset + 2] = 1
    phase[phase_offset + 3] = 1
    phase[phase_offset + 4] = 1
    phase[phase_offset + 5] = 1
    phase[phase_offset + 6] = 1
    phase[phase_offset + 7] = 1
    phase[phase_offset + 8] = 1
    phase[phase_offset + 9] = 1
    phase[phase_offset + 10] = 1
    phase[phase_offset + 11] = 1
    phase[phase_offset + 12] = 1
    phase[phase_offset + 13] = 1
    phase[phase_offset + 14] = 1
    phase[phase_offset + 15] = 1
    phase[phase_offset + 16] = 1
    phase[phase_offset + 17] = 1
    phase[phase_offset + 18] = 1
    phase[phase_offset + 19] = 1
    phase[phase_offset + 20] = 1
    phase[phase_offset + 21] = 1
    phase[phase_offset + 22] = 1
    phase[phase_offset + 23] = 1
    phase[phase_offset + 24] = 1
    phase[phase_offset + 25] = 1
    phase[phase_offset + 26] = 1
    phase[phase_offset + 27] = 1

    # # 数据生成
    # handler = SoundTrackingDataHandler(q_config['in_cut_start'], q_config['q_one'], q_config['lut'],
    #                                    input_size=16, hidden_size=128, sequence_length=39)
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    q_config = handler.qconfig
    gru_data = generate_gru_data(handler, size_y=3, size_x=16)

    clock_in_phase = 1_000
    config = gen_gru_map_config(phase, clock_in_phase=clock_in_phase, size_x=16, size_y=3, data=gru_data,
                                in_data_en=False, out_data_en=False, chip=chip,
                                in_cut_start_mat=q_config['in_cut_start'],
                                q_ones=q_config['q_one'], lut_mat=q_config['lut'], handler=handler)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        del_command = 'rd/s/q cmp_out'
        os.system(del_command)
        os.chdir(c_path)

    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
