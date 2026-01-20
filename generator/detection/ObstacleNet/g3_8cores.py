import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_3_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=None,
                     delay_l4=None, delay_l5=None, axon_delay_empty_phase=2):
    """
        Obstacle: Group 3
        core_x * core_y: 8 * 1
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
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    # ******** 数据交互 ********
    offset = 3 + axon_delay_empty_phase
    # 发送 2*15*128; 接收3 * 63 * 64
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            soma1 = None
            if out_data_en and core_x in [0, 1, 2]:
                soma1 = p06(addr_in=0x9cc0 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=128, num_in=30, length_ciso=1, num_ciso=30, length_out=128,
                            num_out=30, type_in=1, type_out=1,
                            data_in=data['maxpool2']['output'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and core_x in [0, 1, 2], receive_en=in_data_en,
                         send_num=2 * 15 * 128 // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=3 * 63 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=1, X=4 - core_x, Y=1 - core_y, A=480 * (core_x - 0),
                                pack_per_Rhead=2 * 15 * 128 // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=63 * 64, num_in=3, length_ciso=1, num_ciso=3,
                            length_out=63 * 64, num_out=3, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            soma1 = None
            if out_data_en and core_x in [3, 4, 5]:
                soma1 = p06(addr_in=0x9cc0 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=128, num_in=30, length_ciso=1, num_ciso=30, length_out=128,
                            num_out=30, type_in=1, type_out=1,
                            data_in=data['maxpool2']['output'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and core_x in [3, 4, 5], receive_en=in_data_en,
                         send_num=2 * 15 * 128 // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=3 * 63 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=1, X=4 - core_x, Y=1 - core_y, A=480 * (core_x - 3),
                                pack_per_Rhead=2 * 15 * 128 // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x2F40 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=63 * 64, num_in=3, length_ciso=1, num_ciso=3,
                            length_out=63 * 64, num_out=3, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            axon = None
            soma1 = None
            if out_data_en and core_x in [6, 7]:
                soma1 = p06(addr_in=0x9cc0 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=128, num_in=30 if core_x == 6 else 15, length_ciso=1,
                            num_ciso=30 if core_x == 6 else 15, length_out=128,
                            num_out=30 if core_x == 6 else 15, type_in=1, type_out=1,
                            data_in=data['maxpool2']['output'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and core_x in [6, 7], receive_en=in_data_en,
                         send_num=30 * 128 // 8 - 1 if core_x == 6 else 15 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380,
                         addr_din_length=63 * 64 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 64 // 8 - 1,
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=1, X=4 - core_x, Y=1 - core_y, A=480 * (core_x - 6),
                                pack_per_Rhead=30 * 128 // 8 - 1 if core_x == 6 else 15 * 128 // 8 - 1, A_offset=0,
                                Const=0, EN=1)
            soma2 = None
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x5E80 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=63 * 64, num_in=1 if core_x == size_x - 1 else 2, 
                            length_ciso=1, num_ciso=1 if core_x == size_x - 1 else 2,
                            length_out=63 * 64, num_out=1 if core_x == size_x - 1 else 2,
                            type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********
    for core_y, core_x in product(range(size_y), range(size_x)):
        for i in range(axon_delay_empty_phase):
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                   'soma2': None})
    # 放置数据
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            row_num = 7 if core_x == 7 else 8
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x0000 >> 2, length_in=64 * 63,
                        num_in=row_num, length_ciso=1, num_ciso=row_num, length_out=1, num_out=row_num,
                        type_in=1, type_out=1,
                        data_in=data['conv2']['input'][(core_x, core_y)] if (not in_data_en) else None,
                        data_ciso=None)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 1-7发送一行交叠给0-6
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon, soma1 = None, None
            if core_x != 0:
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=63 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 64, num_out=1,
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=False if core_x == 0 else True,
                         receive_en=False if core_x == 7 else True,
                         send_num=63 * 64 // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=63 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if core_x != 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=63 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if core_x != 7:
                soma2 = p06(addr_in=0x8380, addr_out=0x7e00 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=63 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 64, num_out=1,
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv2 + MaxPool + ReLU -- 1/2
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=63, py=7 if core_x == 7 else 9, cin=64, cout=64, kx=3, ky=3, sx=2, sy=2, addr_ina=0x0000 >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x19000 >> 2 if core_x % 2 == 0 else 0x19100 >> 2,
                       addr_out=0x19200 >> 2, ina_type=1, inb_type=1, load_bias=2, pad_top=0, pad_down=0,
                       pad_left=0, pad_right=0, data_x=None,
                       data_w=data['conv2']['weight'][(core_x, core_y)],
                       data_b=data['conv2']['bias1'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19200 >> 2, addr_out=0x8dc0 >> 2, cin=64, cout=64,
                        px=31, py=3 if core_x == 7 else 4,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['conv2'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 1/3  -- 12kB
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x10000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 2/3  -- 12kB
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=(0x10000 + 12 * 1024) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=(0x10000 + 12 * 1024) >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 3/3  -- 12kB
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=(0x10000 + 24 * 1024) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=(0x10000 + 24 * 1024) >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv2 + MaxPool + ReLU -- 2/2
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=63, py=7 if core_x == 7 else 9, cin=64, cout=64, kx=3, ky=3, sx=2, sy=2, addr_ina=0x0000 >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x19000 >> 2 if core_x % 2 == 1 else 0x19100 >> 2,
                       addr_out=0x19200 >> 2, ina_type=1, inb_type=1, load_bias=2, pad_top=0, pad_down=0,
                       pad_left=0, pad_right=0, data_x=None, data_w=None,
                       data_b=data['conv2']['bias2'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19200 >> 2, addr_out=0x9540 >> 2, cin=64, cout=64,
                        px=31, py=3 if core_x == 7 else 4, kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['conv2'],
                        row_ck_on=1, in_row_max=2)
            router = None
            if core_x % 2 == 0:
                addr_f, addr_b = 0x8dc0 >> 2, 0x9540 >> 2
            else:
                addr_f, addr_b = 0x9540 >> 2, 0x8dc0 >> 2
            num_in = 2 * 15 if core_x != 7 else 15
            soma2 = p06(addr_in=addr_f, addr_out=0x9cc0 >> 2, addr_ciso=addr_b,
                        length_in=64, num_in=num_in, length_ciso=64, num_ciso=num_in, length_out=128, num_out=num_in,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 1/3  -- 12kB
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x10000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 2/3  -- 12kB
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=(0x10000 + 12 * 1024) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=(0x10000 + 12 * 1024) >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换权重 3/3  -- 12kB
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=(0x10000 + 24 * 1024) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            delta_x = 1 if core_x % 2 == 0 else -1
            router.addRHead(S=0, T=1, P=0, Q=0, X=delta_x, Y=0, A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=(0x10000 + 24 * 1024) >> 2, addr_ciso=0x0000 >> 2,
                        length_in=256, num_in=48, length_ciso=1, num_ciso=48, length_out=256, num_out=48,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    return map_config


if __name__ == '__main__':
    import os
    from generator.detection.detection_data_handler import DetectionDataHandler
    from generator.detection.ObstacleNet.g3_data import generate_g3_data
    from generator.detection.quantization_config import QuantizationConfig

    case_file_name = 'Obstacle_3'
    chip = (0, 0)
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

    phase[:] = 1

    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g3_data(handler, size_y=1, size_x=8)
    qconfig = QuantizationConfig(name='obstacle')
    in_cut_start_dict = qconfig['in_cut_start']

    clock_in_phase = 150_000
    config = gen_3_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1, data=data,
                              in_data_en=False, out_data_en=False, chip=chip, in_cut_start_dict=in_cut_start_dict)
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
