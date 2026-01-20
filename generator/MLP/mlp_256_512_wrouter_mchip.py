import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_map_config(phase_en, clock_in_phase, size_x, size_y, delay_l4=None, delay_l5=None,
                   in_data_en=True, out_data_en=True, chip=(0, 0)):
    """
        MLP 256-512
        core_x * core_y: 16 * 10
    """
    map_config = {
        'sim_clock': None,
        'step_clock': {
            ((0, 0), 0): (20000 - 1, 40000)
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

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=64, cout=64, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x9000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x0 >> 2,
                       addr_out=0x0000 >> 2, ina_type=1, inb_type=3,
                       load_bias=0, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=[] if not in_data_en else None, data_w=[], data_b=None)
            if out_data_en:
                soma1 = pX5(mode='max', addr_in=0x0000 >> 2, addr_out=0x9000, cin=64, cout=64, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=4, row_ck_on=1,
                            in_row_max=1)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=64 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)

            if in_data_en:
                soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0x9000 >> 2, cin=64, cout=64, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    import copy

    case_file_name = 'MLP_64_64_4chips_40steps'
    config = MapConfigGen()

    phase = np.zeros(50).astype(int)

    phase[:] = 1

    config_0_0 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                                in_data_en=False, out_data_en=True, chip=(0, 0))
    config_0_1 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                                in_data_en=True, out_data_en=True, chip=(0, 0))

    config_1 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                              in_data_en=True, out_data_en=True, chip=(0, 1))

    config_2 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                              in_data_en=True, out_data_en=True, chip=(0, 2))

    config_3_0 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                                in_data_en=True, out_data_en=True, chip=(0, 3))
    config_3_1 = gen_map_config(phase, clock_in_phase=50000, size_x=16, size_y=1,
                                in_data_en=True, out_data_en=False, chip=(0, 3))

    config.add_config(config_0_0, core_offset=(0, 0))
    for i in range(1, 10):
        config.add_config(copy.deepcopy(config_0_1), core_offset=(0, i))
    for i in range(0, 10):
        config.add_config(copy.deepcopy(config_1), core_offset=(0, i))
    for i in range(0, 10):
        config.add_config(copy.deepcopy(config_2), core_offset=(0, i))
    for i in range(0, 9):
        config.add_config(copy.deepcopy(config_3_0), core_offset=(0, i))
    config.add_config(config_3_1, core_offset=(0, 9))

    MapConfigGen.add_router_info(map_config=config.map_config)

    config.map_config['sim_clock'] = 5000 * 40
    config.map_config['step_clock'] = {
        ((0, 0), 0): (5000 - 1, 5000),
        ((0, 1), 0): (5000 - 1, 5000),
        ((0, 2), 0): (5000 - 1, 5000),
        ((0, 3), 0): (5000 - 1, 5000),
    }

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

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test()
