import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, name, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=None,
                     delay_l4=None, delay_l5=None, mouse_net_en=False, axon_delay_empty_phase=2):
    """
        Obstacle: Group 1
        core_x * core_y: 8 * 2
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
    # 接收 33*256*3/1024/2; 发送3 * 63 * 32
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
            if out_data_en:
                soma1 = p06(addr_in=0x6300 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=63 * 32, num_in=3, length_ciso=1, num_ciso=3,
                            length_out=63 * 32, num_out=3, type_in=1, type_out=1,
                            data_in=data['maxpool1']['output1'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            din_length = 32 * 256 * 3 // 2 // 8 if core_x == 7 else 33 * 256 * 3 // 2 // 8
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en,
                         send_num=3 * 63 * 32 // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=din_length - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=din_length - 1, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=3 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=32 // 8, pack_per_Rhead=3 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
            soma2 = None
            if in_data_en:
                if name == 'obstacle':
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 1, 0, 1
                    else:
                        router.CXY, router.Nx, router.Ny = mouse_net_en, -8, 0
                elif name == 'mouse':
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=16, num_in=din_length // 2, length_ciso=1, num_ciso=din_length // 2,
                            length_out=16, num_out=din_length // 2, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收 33*256*3/1024/2; 发送3 * 63 * 32
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
            if out_data_en:
                soma1 = p06(addr_in=0x7AA0 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=63 * 32, num_in=3, length_ciso=1, num_ciso=3,
                            length_out=63 * 32, num_out=3, type_in=1, type_out=1,
                            data_in=data['maxpool1']['output2'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            din_length = 32 * 256 * 3 // 2 // 8 if core_x == 7 else 33 * 256 * 3 // 2 // 8
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en,
                         send_num=3 * 63 * 32 // 8 - 1, receive_num=0, addr_din_base=0x380,
                         addr_din_length=din_length - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=din_length - 1, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=3 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=32 // 8, pack_per_Rhead=3 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
            soma2 = None
            if in_data_en:
                if name == 'obstacle':
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 1, 0, 1
                    else:
                        router.CXY, router.Nx, router.Ny = mouse_net_en, -8, 0
                elif name == 'mouse':
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
                soma2 = p06(addr_in=0x8380, addr_out=(0x0000 + din_length * 8) >> 2, addr_ciso=0x0000 >> 2,
                            length_in=16, num_in=din_length // 2, length_ciso=1, num_ciso=din_length // 2,
                            length_out=16, num_out=din_length // 2, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送2(1) * 63 * 32
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                       axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if not (core_x, core_y) == (0, 0):
                axon = None
            axon = None
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0x9240 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=63 * 32, num_in=1 if core_x == size_x - 1 else 2,
                            length_ciso=1, num_ciso=1 if core_x == size_x - 1 else 2,
                            length_out=63 * 32, num_out=1 if core_x == size_x - 1 else 2,
                            type_in=1, type_out=1,
                            data_in=data['maxpool1']['output3'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=False,
                         send_num=63 * 32 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 32 // 8 - 1,
                         receive_num=0, addr_din_base=0x0,
                         addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0,
                                    pack_per_Rhead=63 * 32 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=32 // 8,
                                    pack_per_Rhead=63 * 32 // 8 - 1 if core_x == size_x - 1 else 2 * 63 * 32 // 8 - 1,
                                    A_offset=32 // 8, Const=32 // 8 - 1, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********
    for core_y, core_x in product(range(size_y), range(size_x)):
        for i in range(axon_delay_empty_phase):
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                   'soma2': None})
    # Conv计算
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=256, py=32 if core_x == size_x - 1 else 33, cin=3, cout=32, kx=3, ky=3, sx=2, sy=2,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10360 >> 2, addr_out=0x103E0 >> 2,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_x=None if in_data_en else data['conv1']['input'][(core_x, core_y)],
                       data_w=data['conv1']['weight'][(core_x, core_y)],
                       data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x103E0 >> 2, addr_out=0x6300 >> 2,
                        cin=32, cout=32, px=127, py=15 if core_x == size_x - 1 else 16,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['conv1'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

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
