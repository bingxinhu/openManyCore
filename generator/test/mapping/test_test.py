import pickle
import os
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
import json
import numpy as np


# test_mapping编号

# IR_L1L2.xmk
tb_name = 'test'


def read_ir(name: str):
    config = None
    with open("temp/mapping_ir/ws_wcz/"+name, "rb") as f:
        config = pickle.load(f)
    return config

#


def test_mapping():
    map_config = {}
    map_config[((1, 1), 0)] = {}
    map_config[((0, 0), 0)] = {}

    ws = read_ir("ir_chiptrans_receive_mul_renew.mapping")

    for step_id, group_id, config in MapConfig(ws):
        phase_num = None
        if isinstance(step_id, str):
            continue
        for id in config.core_list:
            config._core_cofig[id]["prims"] = config._core_cofig[id]["prims"][0:1]
    map_config[((1, 0), 0)] = {}
    map_config[((1, 0), 0)][0] = ws[0][0]
    map_config[((1, 0), 0)][1] = ws[0][1]
    map_config[((1, 1), 0)] = {}
    map_config[((1, 1), 0)][0] = ws[0][2]
    map_config[((0, 1), 0)] = {}
    map_config[((0, 1), 0)][0] = ws[0][3]
    map_config[((0, 0), 0)][0] = ws[0][4]

    wcz = read_ir("Chip11_ir_wcz3333.map")
    for j in range(7):
        # if j == 0:                        #ok (14,0)
        #     continue
        # if j == 1 or j == 0:              # ok (14,0)+(14,1)
        #     continue
        # if j == 0 or j == 2:              # ok (14,0)+(14,2)
        #     continue
        # if j == 1:                        # ok (14,1)
        #     continue
        # if j == 1 or j == 2:              # ok (14,1)+(14,2)
        #     continue
        if j == 1 or j == 0 or j == 2:    # (14,0)+(14,1)+(14,2)
            continue
        wcz[((1, 1), 0)][0].pop(((1, 1), (14, j)))
    for i in range(14):
        # if i == 1 or i == 0:                      #ok (14,0)
        #     continue
        # if i == 2 or i == 3 or i == 1 or i == 0:  # ok (14,0)+(14,1)
        #     continue
        # if i == 1 or i == 0 or i == 4 or i == 5:  # ok (14,0)+(14,2)
        #     continue
        # if i == 2 or i == 3:                      # ok (14,1)
        #     continue
        # if i == 2 or i == 3 or i == 4 or i == 5:  # ok (14,1)+(14,2)
        #     continue
        # (14,0)+(14,1)+(14,2)
        if i == 2 or i == 3 or i == 1 or i == 0 or i == 4 or i == 5:
            continue
        wcz[((1, 1), 0)][1].pop(((1, 1), (i, 0)))
        wcz[((1, 1), 0)][1].pop(((1, 1), (i, 1)))
        wcz[((1, 1), 0)][4].pop(((1, 1), (i, 2)))

    # wcz[((1, 1), 0)][0]['clock'] = 80000
    # wcz[((1, 1), 0)][1]['clock'] = 80000
    # wcz[((1, 1), 0)][4]['clock'] = 80000

    map_config[((1, 1), 0)][1] = wcz[((1, 1), 0)][0]
    map_config[((1, 1), 0)][2] = wcz[((1, 1), 0)][1]
    map_config[((1, 1), 0)][3] = wcz[((1, 1), 0)][4]

    # wcz = read_ir("Chip1_1_ir_wcz.mapping")
    # map_config[((1, 1), 0)][1] = wcz[((1, 1), 0)][0]
    # map_config[((1, 1), 0)][2] = wcz[((1, 1), 0)][1]
    # map_config[((1, 1), 0)][3] = wcz[((1, 1), 0)][2]
    # map_config[((1, 1), 0)][5] = wcz[((1, 1), 0)][4]
    # map_config[((1, 1), 0)][6] = wcz[((1, 1), 0)][5]
    # map_config[((1, 1), 0)][7] = wcz[((1, 1), 0)][6]

    with open('mapping_ir_merged.mapping', 'wb') as f:
        pickle.dump(map_config, f)
    # map_config['sim_clock'] = 10000
    test_phase = []
    for step_id, group_id, config in MapConfig(map_config):
        phase_num = None
        if not isinstance(config, GroupConfig):
            continue
        for id in config.core_list:
            phase_num = len(config.axon_list(id))
            if phase_num is not None:
                assert phase_num == len(config.axon_list(id))
        for i in range(phase_num):
            test_phase.append((group_id, i+1))  # 确保每个group的所有core执行相同的phase数

    from generator.test_engine.test_config import HardwareDebugFileSwitch
    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.dict,
        'test_group_phase': test_phase
    }
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


test_mapping()
