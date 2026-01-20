import pickle
import os
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
import json
import numpy as np


# test_mapping编号

# IR_L1L2.xmk
tb_name = 'L11999'


def read_ir(name: str):
    config = None
    with open("temp/mapping_ir/mchip/"+name, "rb") as f:
        config = pickle.load(f)
    return config

#


def test_mapping():
    map_config = {}

    xmk_snn_1core = read_ir("IR_snn_1core.xmk")
    map_config[((0, 0), 0)] = {}
    map_config[((0, 0), 0)][0] = xmk_snn_1core[0][0]

    ws = read_ir("ir_chiptrans_receive_mul_renew.mapping")
    map_config[((1, 0), 0)] = {}
    map_config[((1, 0), 0)][0] = ws[0][0]
    map_config[((1, 0), 0)][1] = ws[0][1]
    map_config[((1, 1), 0)] = {}
    map_config[((1, 1), 0)][0] = ws[0][2]
    map_config[((0, 1), 0)] = {}
    map_config[((0, 1), 0)][0] = ws[0][3]

    xmk_IR_3group_chip = read_ir("IR_3group_chip.xmk")
    map_config[((0, 0), 0)][1] = xmk_IR_3group_chip[((0, 0), 0)][0]
    map_config[((1, 0), 0)][2] = xmk_IR_3group_chip[((1, 0), 0)][1]
    map_config[((1, 0), 0)][3] = xmk_IR_3group_chip[((1, 0), 0)][2]

    lms_lstm_1core = read_ir("IR_lstm_1core.lms")
    map_config[((0, 0), 0)][2] = lms_lstm_1core[0][0]

    wcz = read_ir("Chip1_1_ir_wcz.mapping")
    # map_config = wcz
    # map_config[((1, 1), 0)] = {}
    # map_config[((1, 1), 0)][1] = wcz[((1, 1), 0)][0]
    # map_config[((1, 1), 0)][2] = wcz[((1, 1), 0)][1]
    # map_config[((1, 1), 0)][3] = wcz[((1, 1), 0)][2]
    map_config[((1, 1), 0)][4] = wcz[((1, 1), 0)][3]
    # map_config[((1, 1), 0)][5] = wcz[((1, 1), 0)][4]
    # map_config[((1, 1), 0)][6] = wcz[((1, 1), 0)][5]
    # map_config[((1, 1), 0)][7] = wcz[((1, 1), 0)][6]
    map_config[((1, 2), 0)] = {}
    map_config[((1, 2), 0)][0] = wcz[((1, 2), 0)][0]

    # sch_0717_phase0-phase4_2104.map_config
    # 0-6   1,2
    # 7     1,1
    # 8     2,2
    # 9-13  2,1
    # 14-16 2,0
    sch = read_ir("sch_0717_phase0-phase4_2104.map_config")
    # map_config=sch
    for i in range(7):
        map_config[((1, 2), 0)][i+1] = sch[0][i]

    map_config[((1, 1), 0)][8] = sch[0][7]

    map_config[((2, 2), 0)] = {}
    map_config[((2, 2), 0)][0] = sch[0][8]

    map_config[((2, 1), 0)] = {}
    for i in range(9, 14):
        map_config[((2, 1), 0)][i-9] = sch[0][i]

    map_config[((2, 0), 0)] = {}
    for i in range(14, 17):
        map_config[((2, 0), 0)][i-14] = sch[0][i]

    map_config[((1, 0), 0)][4] = sch[0][17]

    # map_config = xmk_IR_3group_chip
    # for y in range(0, 2):
    #     for x in range(14, 16):
    #         map_config[0][2].pop(((0, 0), (x, y)), None)

    # for step_id, group_id, config in MapConfig(map_config):
    #     phase_num = None
    #     if isinstance(step_id, str):
    #         continue
    #     for id in config.core_list:
    #         config._core_cofig[id]["prims"] = config._core_cofig[id]["prims"][0:1]

    map_config['sim_clock'] = 300000

    with open('mapping_ir_merged.mapping', 'wb') as f:
        pickle.dump(map_config, f)

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
