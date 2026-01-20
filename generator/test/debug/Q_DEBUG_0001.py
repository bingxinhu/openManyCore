import numpy as np
import sys
import os
import os

sys.path.append(os.getcwd())
from itertools import product
from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen

"""
    ResNet-50 5-Chip Group-15 phase 4 行流水问题
"""
map_config = {
    'sim_clock': None,
    ((0, 0), 0): {
        0: {
            'clock': 100_000,
            'mode': 1,
            ((0, 0), (0, 0)): {
                'prims': []
            }
        }
    }
}

phase_group = map_config[((0, 0), 0)][0]

# L41 部分和求和，流水ReLU
for core_y, core_x in product(range(1), range(1)):
    # if core_y in [0]:
    #     px, py = 1, 13      # px, py = 1, 13 FIXME 这个行流水的时候结果有问题
    # else:
    #     px, py = 4, 3
    px, py = 13, 2
    axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
               addr_in=0x19000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
               pad_right=0, bias_length=0, data_x=[], data_b=None, constant_b=0)
    soma1 = pX5(mode='max', addr_in=0x0000 >> 2, addr_out=0xfe60 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=1,
                row_ck_on=1, in_row_max=1)
    router = None
    soma2 = None
    phase_group[((0, 0), (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                             'soma2': soma2})

    case_file_name = 'Q_DEBUG_0016'

    from generator.resnet50.resnet50_5chips.G15_data import generate_g15_data

    config = map_config
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = 100_000

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
