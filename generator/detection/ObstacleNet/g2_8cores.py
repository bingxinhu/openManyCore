import numpy as np
import sys
import os

from numpy import core
from numpy.lib.arraypad import pad

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_2_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False,
                     delay_l4=None, delay_l5=None, axon_delay_empty_phase=2):
    """
        Obstacle: Group 2
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
    # 接收/发送3 * 63 * 64
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x6E40 >> 2 if core_x == size_x - 1 else 0x7E00 >> 2, 
                            addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, 
                            length_in=63 * 64, num_in=3, 
                            length_ciso=1, num_ciso=3, 
                            length_out=63 * 64, num_out=3, 
                            type_in=1, type_out=1, 
                            data_in=data['res1']['add']['output1'][(core_x, core_y)] if init_data else None, 
                            data_ciso=None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, 
                         send_num=3 * 63 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x1000 >> 2, addr_din_length=3 * 63 * 64 // 8 - 1, 
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=3 * 63 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, 
                            length_in=63 * 64, num_in=3, length_ciso=1, num_ciso=3, length_out=63 * 64, num_out=3, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收/发送3 * 63 * 64
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x9D80 >> 2 if core_x == size_x - 1 else 0xAD40 >> 2, 
                            addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, 
                            length_in=63 * 64, num_in=3, 
                            length_ciso=1, num_ciso=3, 
                            length_out=63 * 64, num_out=3, 
                            type_in=1, type_out=1, 
                            data_in=data['res1']['add']['output2'][(core_x, core_y)] if init_data else None, 
                            data_ciso=None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, 
                         send_num=3 * 63 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x1000 >> 2, addr_din_length=3 * 63 * 64 // 8 - 1, 
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=3 * 63 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2F40 >> 2, addr_ciso=0x10000 >> 2, 
                            length_in=63 * 64, num_in=3, length_ciso=1, num_ciso=3, length_out=63 * 64, num_out=3, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收/发送2(1) * 63 * 64
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
            if out_data_en:
                soma1 = p06(addr_in=0xCCC0 >> 2 if core_x == size_x - 1 else 0xDC80 >> 2, 
                            addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, 
                            length_in=63 * 64, num_in=1 if core_x == size_x - 1 else 2, 
                            length_ciso=1, num_ciso=1 if core_x == size_x - 1 else 2, 
                            length_out=63 * 64, num_out=1 if core_x == size_x - 1 else 2, 
                            type_in=1, type_out=1, 
                            data_in=data['res1']['add']['output3'][(core_x, core_y)] if init_data else None, 
                            data_ciso=None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, 
                         send_num=63 * 64 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x1000 >> 2, 
                         addr_din_length=63 * 64 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 64 // 8 - 1, 
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, 
                                pack_per_Rhead=63 * 64 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x5E80 >> 2, addr_ciso=0x10000 >> 2, 
                            length_in=63 * 64, num_in=1 if core_x == size_x - 1 else 2,
                            length_ciso=1, num_ciso=1 if core_x == size_x - 1 else 2, 
                            length_out=63 * 64, num_out=1 if core_x == size_x - 1 else 2, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # ******** 开始计算 ********
    for core_y, core_x in product(range(size_y), range(size_x)):
        for i in range(axon_delay_empty_phase):
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                   'soma2': None})
    # Conv1
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=63, py=7 if core_x == size_x - 1 else 8, cin=64, cout=32, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10800 >> 2, addr_out=0x7E00 >> 2, 
                       ina_type=1, inb_type=1, load_bias=2, 
                       data_x=None if in_data_en else data['res1']['conv1']['input'][(core_x, core_y)], 
                       data_w=data['res1']['conv1']['weight'][(core_x, core_y)],
                       data_b=data['res1']['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x7E00 >> 2, addr_out=0x15960 >> 2,
                        cin=32, cout=32, px=63, py=7 if core_x == size_x - 1 else 8,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, 
                        in_cut_start=in_cut_start_dict['res1']['conv1'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送交叠1
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_x == size_x - 1:
                soma1 = None
            else:
                soma1 = p06(addr_in=0x19080 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, 
                            length_in=63 * 32, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 32, num_out=1, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            send_en = False if core_x == size_x - 1 else True
            receive_en = False if core_x == 0 else True
            router = p09(rhead_mode=1, send_en=send_en, receive_en=receive_en, 
                         send_num=63 * 32 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=63 * 32 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if send_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=63 * 32 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if receive_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x15180 >> 2, addr_ciso=0x0000 >> 2, 
                            length_in=63 * 32, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 32, num_out=1, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送交叠2
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_x == 0:
                soma1 = None
            else:
                soma1 = p06(addr_in=0x15960 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, 
                            length_in=63 * 32, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 32, num_out=1, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            send_en = False if core_x == 0 else True
            receive_en = False if core_x == size_x - 1 else True
            router = p09(rhead_mode=1, send_en=send_en, receive_en=receive_en, 
                         send_num=63 * 32 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=63 * 32 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if send_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=63 * 32 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if receive_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x19860 >> 2, addr_ciso=0x0000 >> 2, 
                            length_in=63 * 32, num_in=1, length_ciso=1, num_ciso=1, length_out=63 * 32, num_out=1, 
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv2
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == 0:
                py = 9
                addr_ina = 0x15960 >> 2
                pad_top = 1
                pad_down = 0
                addr_out = 0x7E00 >> 2
            elif core_x == size_x - 1:
                py = 8
                addr_ina = 0x15180 >> 2
                pad_top = 0
                pad_down = 1
                addr_out = 0x6E40 >> 2
            else:
                py = 10
                addr_ina = 0x15180 >> 2
                pad_top = 0
                pad_down = 0
                addr_out = 0x7E00 >> 2
            axon = p41(px=63, py=py, cin=32, cout=64, kx=3, ky=3, sx=1, sy=1,
                       addr_ina=addr_ina, addr_inb=0x10880 >> 2, addr_bias=0x15080 >> 2, addr_out=0x1A040 >> 2, 
                       pad_top=pad_top, pad_down=pad_down, pad_left=1, pad_right=1,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None, 
                       data_w=data['res1']['conv2']['weight'][(core_x, core_y)],
                       data_b=data['res1']['conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1A040 >> 2, addr_out=addr_out,
                        cin=64, cout=64, px=63, py=7 if core_x == size_x - 1 else 8,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, 
                        in_cut_start=in_cut_start_dict['res1']['conv2'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Add
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=64, 
                       px=63, py=7 if core_x == size_x - 1 else 8,
                       addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x15180 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x15180 >> 2, 
                        addr_out=0x6E40 >> 2 if core_x == size_x - 1 else 0x7E00 >> 2,
                        cin=64, cout=64, px=63, py=7 if core_x == size_x - 1 else 8,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['res1']['add'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})        

    return map_config


if __name__ == '__main__':
    import os
    from generator.detection.detection_data_handler import DetectionDataHandler
    from generator.detection.ObstacleNet.g2_data import generate_g2_data
    from generator.detection.quantization_config import QuantizationConfig

    case_file_name = 'Obstacle_2'
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

    # phase[14:] = 0

    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g2_data(handler, size_y=1, size_x=8)
    qconfig = QuantizationConfig(name='obstacle')
    in_cut_start_dict = qconfig['in_cut_start']

    clock_in_phase = 150_000
    config = gen_2_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1, data=data,
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
