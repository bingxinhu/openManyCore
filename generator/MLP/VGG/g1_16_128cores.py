import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, delay_l4=None, delay_l5=None,
                     in_data_en=True, out_data_en=True, chip=(0, 0)):
    """
        VGG MLP: Group 1
        core_x * core_y: 16 * 1
    """
    map_config = {
        'sim_clock': None,
        'step_clock': {
            ((0, 0), 0): (3500, 10000)
        },
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
    offset = 1
    # 接收1568
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x8000 >> 2, addr_out=0x24000 >> 2,
                        addr_ciso=0x0000, length_in=32,
                        length_out=32, length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=1, row_ck_on=0,
                        data_in=[], data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=1568 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=1568 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=1568 // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2,
                        addr_ciso=0x10000 >> 2, length_in=32,
                        length_out=32, length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                        data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # Linear
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=1568, cout=32, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x1C400 >> 2,
                       addr_out=0x0620 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None, data_w=[], data_b=[])
            soma1 = p06(addr_in=0x0620 >> 2, addr_out=0x24000 >> 2,
                        addr_ciso=0x0000, length_in=32,
                        length_out=32, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=0,
                        type_out=0, in_cut_start=0, in_row_max=1, row_ck_on=1,
                        data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=False if core_x == 0 else True,
                         receive_en=True if core_x == 0 else False, send_num=4 * 32 // 8 - 1, receive_num=15 - 1,
                         addr_din_base=0x1000 >> 2, addr_din_length=15 * 4 * 32 // 8 - 1, 
                         addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if core_x != 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-core_x, Y=0, A=4 * 32 // 8 * (core_x - 1), 
                                pack_per_Rhead=4 * 32 // 8 - 1, 
                                A_offset=0, Const=0, EN=1)
            if core_x == 0:
                soma1 = None
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x06A0 >> 2,
                            addr_ciso=0x0000 >> 2, length_in=15 * 32,
                            length_out=15 * 32, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=0,
                            type_out=0, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=16, sx=1, sy=1, cin=32, px=1, py=1,
                       addr_in=0x0620 >> 2, addr_bias=0x0, addr_out=0x1C480 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x1C480 >> 2, addr_out=0x0E20 >> 2,
                        cin=32, cout=32, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=0, row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'VGG_MLP_1'
    chip = (0, 0)

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[:] = 1

    clock_in_phase = 10000
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=16, size_y=10,
                              in_data_en=True, out_data_en=True, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)
    config['sim_clock'] = 3500

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
