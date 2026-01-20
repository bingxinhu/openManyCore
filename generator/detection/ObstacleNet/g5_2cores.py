import numpy as np
import sys
import os

from numpy import core
from numpy.lib.arraypad import pad

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_5_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=None,
                     delay_l4=None, delay_l5=None, axon_delay_empty_phase=2):
    """
        Obstacle: Group 5
        core_x * core_y: 2 * 1
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
    offset = 1 + axon_delay_empty_phase
    # 发送 6*6*128;  接收 6*6*128
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            axon = None
            soma1 = None
            if out_data_en and core_x == 0:
                soma1 = p06(addr_in=0xc400 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=128, num_in=36, length_ciso=1, num_ciso=36, length_out=128,
                            num_out=36, type_in=1, type_out=1,
                            data_in=data['res3']['add']['output'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and core_x == 0, receive_en=in_data_en,
                         send_num=6 * 6 * 128 // 8 - 1, receive_num=3, addr_din_base=0x380,
                         addr_din_length=6 * 6 * 128 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=6 * 6 * 128 // 8 - 1, nx=-1, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=6 * 6 * 128 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en:
                router.CXY = 1 if core_x == 1 else 0
                soma2 = p06(addr_in=0x8380, addr_out=0xB200 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=128, num_in=36, length_ciso=1, num_ciso=36,
                            length_out=128, num_out=36, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********
    for core_y, core_x in product(range(size_y), range(size_x)):
        for i in range(axon_delay_empty_phase):
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                   'soma2': None})
    # res2
    # Conv1
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=6, py=6, cin=128, cout=64, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0xB200 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x12000 >> 2, addr_out=0xC400 >> 2,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_x=None if in_data_en else data['res2']['conv1']['input'][(core_x, core_y)],
                       data_w=data['res2']['conv1']['weight'][(core_x, core_y)],
                       data_b=data['res2']['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0xC400 >> 2, addr_out=0x1B200 >> 2,
                        cin=64, cout=64, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res2']['conv1'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv2
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=6, py=6, cin=64, cout=64, kx=3, ky=3, sx=1, sy=1,
                       addr_ina=0x1B200 >> 2, addr_inb=0x2100 >> 2, addr_bias=0xB100 >> 2, addr_out=0xC400 >> 2,
                       pad_top=1, pad_down=1, pad_left=1, pad_right=1,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['res2']['conv2']['weight'][(core_x, core_y)],
                       data_b=data['res2']['conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0xC400 >> 2, addr_out=0x1BB00 >> 2,
                        cin=64, cout=64, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res2']['conv2'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 互相发送Conv2计算结果
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x1BB00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2,
                        length_in=6 * 6 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=6 * 6 * 64, num_out=1,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=True, receive_en=True,
                         send_num=6 * 6 * 64 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=6 * 6 * 64 // 8 - 1,
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1 if core_x == 0 else -1, Y=0, A=0,
                            pack_per_Rhead=6 * 6 * 64 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x21000 >> 2 if core_x == 1 else 0x1BB00 >> 2, addr_out=0xC400 >> 2,
                        addr_ciso=0x21000 >> 2 if core_x == 0 else 0x1BB00 >> 2,
                        length_in=64, num_in=6 * 6, length_ciso=64, num_ciso=6 * 6, length_out=128, num_out=6 * 6,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Add
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128,
                       px=6, py=6, addr_in=0xB200 >> 2, addr_bias=0x0, addr_out=0xD600 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0, bias_length=0, data_x=None, data_b=None,
                       constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xD600 >> 2, addr_out=0x1B200 >> 2,
                        cin=128, cout=128, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res2']['add'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # res3
    # Conv1
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=6, py=6, cin=128, cout=64, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1B200 >> 2, addr_inb=0x0000 >> 2, addr_bias=0x2000 >> 2, addr_out=0x1C400 >> 2,
                       ina_type=1, inb_type=1, load_bias=2, data_x=data['res3']['conv1']['input'][(core_x, core_y)],
                       data_w=data['res3']['conv1']['weight'][(core_x, core_y)],
                       data_b=data['res3']['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1C400 >> 2, addr_out=0xB200 >> 2,
                        cin=64, cout=64, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res3']['conv1'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv2
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=6, py=6, cin=64, cout=64, kx=3, ky=3, sx=1, sy=1,
                       addr_ina=0xB200 >> 2, addr_inb=0x12100 >> 2, addr_bias=0x1B100 >> 2, addr_out=0x1C400 >> 2,
                       pad_top=1, pad_down=1, pad_left=1, pad_right=1,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['res3']['conv2']['weight'][(core_x, core_y)],
                       data_b=data['res3']['conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1C400 >> 2, addr_out=0xBB00 >> 2,
                        cin=64, cout=64, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res3']['conv2'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 互相发送Conv2计算结果
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0xBB00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2,
                        length_in=6 * 6 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=6 * 6 * 64, num_out=1,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=True, receive_en=True,
                         send_num=6 * 6 * 64 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=6 * 6 * 64 // 8 - 1,
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1 if core_x == 0 else -1, Y=0, A=0,
                            pack_per_Rhead=6 * 6 * 64 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x21000 >> 2 if core_x == 1 else 0xBB00 >> 2, addr_out=0x1C400 >> 2,
                        addr_ciso=0x21000 >> 2 if core_x == 0 else 0xBB00 >> 2,
                        length_in=64, num_in=6 * 6, length_ciso=64, num_ciso=6 * 6, length_out=128, num_out=6 * 6,
                        type_in=1, type_out=1, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Add
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128,
                       px=6, py=6, addr_in=0x1B200 >> 2, addr_bias=0x0, addr_out=0x1D600 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0, bias_length=0, data_x=None, data_b=None,
                       constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x1D600 >> 2, addr_out=0xC400 >> 2,
                        cin=128, cout=128, px=6, py=6,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res3']['add'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.detection.detection_data_handler import DetectionDataHandler
    from generator.detection.ObstacleNet.g5_data import generate_g5_data
    from generator.detection.quantization_config import QuantizationConfig

    case_file_name = 'Obstacle_5'
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
    phase[phase_offset + 5] = 1
    phase[phase_offset + 6] = 1
    phase[phase_offset + 7] = 1

    # phase[14:] = 0

    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g5_data(handler, size_y=1, size_x=2)
    qconfig = QuantizationConfig(name='obstacle')
    in_cut_start_dict = qconfig['in_cut_start']

    clock_in_phase = 150_000
    config = gen_5_map_config(phase, clock_in_phase=clock_in_phase, size_x=2, size_y=1, data=data,
                              in_data_en=False, out_data_en=False, chip=chip, in_cut_start_dict=in_cut_start_dict)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
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
