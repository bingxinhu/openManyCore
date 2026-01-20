import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_0_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, chip=(0, 0),
                     delay_l4=None, delay_l5=None, out_data_en=False, mouse_en=False, obstacle_en=False,
                     send_to_fpga=False, g6_en=False, axon_delay_empty_phase=2):
    """
        Obstacle: Group 0
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
    offset = 0 + axon_delay_empty_phase
    # ******** 开始计算 ********

    # 发送 1584 个包 - 33*256*3/1024/2; 接受 32 * 4 B
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=(0x10000 + 32) >> 2, addr_bias=(0x10000 + 64) >> 2, addr_out=(0x10000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            length = 32 * 256 * 3 // 2 if core_x == 7 else 33 * 256 * 3 // 2
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=16, num_in=length // 16, length_ciso=1, num_ciso=6, length_out=16,
                            num_out=length // 16, type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            if mouse_en and obstacle_en:
                in_data_en = True if g6_en else False
                addr_din_length = (32 + 16) * 4 // 8 - 1
                receive_num = 1
            elif mouse_en or obstacle_en:
                in_data_en = True if g6_en else False
                addr_din_length = 32 * 4 // 8 - 1
                receive_num = 0
            else:
                in_data_en = False
                addr_din_length = 0
                receive_num = 0
            if send_to_fpga:
                in_data_en = False
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en if core_x == 0 else 0,
                         send_num=length // 8 - 1, receive_num=receive_num, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=1, X=0, Y=1, A=0, pack_per_Rhead=length // 8 - 1, A_offset=0, Const=0,
                                EN=1)
            # 放置数据
            soma2 = p06(addr_in=0x0000 >> 2, addr_out=0x8900, addr_ciso=0x0000 >> 2,
                        length_in=256 * 3, num_in=32 if core_x == 7 else 33, length_ciso=1,
                        num_ciso=1, length_out=1, num_out=1, type_in=1, type_out=1,
                        data_in=data['conv1']['input'][(core_x, core_y)],
                        data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 1584 个包 - 33*256*3/1024/2
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=(0x10000 + 32) >> 2, addr_bias=(0x10000 + 64) >> 2, addr_out=(0x10000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            length = 32 * 256 * 3 // 2 if core_x == 7 else 33 * 256 * 3 // 2
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=(0x0000 + length) >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=16, num_in=length // 16, length_ciso=1, num_ciso=6, length_out=16,
                            num_out=length // 16, type_in=1, type_out=1, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0,
                         send_num=length // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=32 * 4 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=1, X=0, Y=1, A=0, pack_per_Rhead=length // 8 - 1, A_offset=0, Const=0,
                                EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    for core_y, core_x in product(range(size_y), range(size_x)):
        for i in range(axon_delay_empty_phase):
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                   'soma2': None})
    return map_config


if __name__ == '__main__':
    import os
    from generator.detection.detection_data_handler import DetectionDataHandler
    from generator.detection.ObstacleNet.g1_data import generate_g1_data
    from generator.detection.quantization_config import QuantizationConfig

    case_file_name = 'Obstacle_1'
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    # phase[0] = 1
    # phase[1] = 1

    phase[phase_offset + 0] = 1

    # phase[14:] = 0

    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g1_data(handler, size_y=2, size_x=8)
    qconfig = QuantizationConfig(name='obstacle')
    in_cut_start_dict = qconfig['in_cut_start']

    clock_in_phase = 150_000
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=2, data=data,
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
