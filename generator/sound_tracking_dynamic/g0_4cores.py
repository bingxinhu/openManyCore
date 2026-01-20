import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.sound_tracking.utils import get_core_id


def gen_0_map_config(phase_en, clock_in_phase, size_x, size_y, cuts: None, data=None, in_data_en=False,
                     out_data_en=False, chip=(0, 0), delay_l4=None, delay_l5=None, send_to_fpga=False):
    """
        Sound Tracking: Group 0
        core_x * core_y: 4 * 1
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
    # 发送 10240B		10240B		 10240B		 11264B; 接收 16B
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0 and core_x == 0:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                        addr_inb=0x10000 >> 2, addr_bias=0x10000 >> 2, addr_out=0x10000 >> 2, axon_delay=True,
                        L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            else:
                axon = None
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            num_in = 11264 // 16 if core_x == 3 else 10240 // 16
            if out_data_en:
                soma1 = p06(addr_in=(0x0000 + 0) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                            length_out=16, length_ciso=1, num_in=num_in, num_ciso=num_in, num_out=num_in, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['fpga']['input1'][(core_x, core_y)],
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en and core_x == 0 and not send_to_fpga,
                         send_num=num_in * 2 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-12, Y=0, A=0, pack_per_Rhead=num_in * 2 - 1, A_offset=0,
                                Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 10240B		10240B		 10240B		 11264B
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == 0 and core_y == 0:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                        addr_inb=0x10000 >> 2, addr_bias=0x10000 >> 2, addr_out=0x10000 >> 2, axon_delay=True,
                        L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            else:
                axon = None
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            num_in = 11264 // 16 if core_x == 3 else 10240 // 16
            if out_data_en:
                soma1 = p06(addr_in=(0x0000 + num_in * 16) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                            length_out=16, length_ciso=1, num_in=num_in, num_ciso=num_in, num_out=num_in, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['fpga']['input2'][(core_x, core_y)],
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0,
                         send_num=num_in * 2 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-12, Y=0, A=0, pack_per_Rhead=num_in * 2 - 1, A_offset=0,
                                Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 10240B		10240B		 10240B		 11264BB
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == 0 and core_y == 0:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                        addr_inb=0x10000 >> 2, addr_bias=0x10000 >> 2, addr_out=0x10000 >> 2, axon_delay=True,
                        L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            else:
                axon = None
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            num_in = 11264 // 16 if core_x == 3 else 10240 // 16
            if out_data_en:
                soma1 = p06(addr_in=(0x0000 + num_in * 32) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                            length_out=16, length_ciso=1, num_in=num_in, num_ciso=num_in, num_out=num_in, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['fpga']['input3'][(core_x, core_y)],
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0,
                         send_num=num_in * 2 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-12, Y=0, A=0, pack_per_Rhead=num_in * 2 - 1, A_offset=0,
                                Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 10400B		 10400B 		 10400B 		 11440B
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            num_in = 11440 // 16 if core_x == 3 else 10400 // 16
            last_num_in = 11264 // 16 if core_x == 3 else 10240 // 16
            if out_data_en:
                soma1 = p06(addr_in=(0x0000 + last_num_in * 48) >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2,
                            length_in=16,
                            length_out=16, length_ciso=1, num_in=num_in, num_ciso=num_in, num_out=num_in, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['fpga']['input4'][(core_x, core_y)],
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0,
                         send_num=num_in * 2 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-12, Y=0, A=0, pack_per_Rhead=num_in * 2 - 1, A_offset=0,
                                Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
    from generator.sound_tracking.g1_data import generate_g1_data
    from generator.sound_tracking.quantization_config import QuantizationConfig

    case_file_name = 'ST_1'
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

    # phase[14:] = 0

    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_g1_data(handler, size_y=1, size_x=4)
    para = QuantizationConfig(sequence_length=39)

    clock_in_phase = 150_000
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1, data=data,
                              in_data_en=False, out_data_en=False, chip=chip, cuts=para['conv1'])
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
