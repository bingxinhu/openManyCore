import pickle
import os
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
import json
import numpy as np


# test_mapping编号
# IR_Phase4_3coreV1.xmk                 ：M11006
# IR_Source_9core_Grp5_test1.mapping    ：M11007
# IR_Source_9core_Grp5_test2.mapping    ：M11008
# IR_Phase6_3coreV1.xmk                 ：M11009
# IR_Phase6_6core.xmk                   : M11010
# IR_Phase6_9core.xmk                   : M11011
# IR_20Phase_9core.xmk                  ：M11012 (前5个phase)
# IR_20Phase_9core.xmk                  ：M11013 (前9个phase)
# IR_phase9_10_6core.xmk                : M11014（9、10phase）
# IR_20Phase_9core.xmk                  ：M11015 (前10个phase)
# IR_Source_9core_Grp3.mapping          : M11016
# IR_phase14_17_6core.xmk               ：M11021
# IR_Source_3x3core_Grp3.mapping        : M11022
# IR_Source_28core_Grp3_4Phase.mapping  ：M11023
# G15_64cores_phase24_28.map_config     : M11024
# IR_20phase_9core_new.xmk              ：M11025
# IR_Source_28core_Grp3_4Phase.mapping  ：M11026
# IR_Source_28core_Grp3_4.mapping       ：M11028
# IR_lstm.lms                           ：M11029
# IR_Source_28core_Grp3_4Phasepaotong.mapping   : S11035
# ir_Grp2_1to10.mapping                 ：S11040
# ir_mem2error.mapping
# IR_28core_L1.xmk
# 归回-----------
# test_sch_0709_pass.map_config
# test_sch_0709_nopass.map_config
# test_sch_0710.map_config
# test_sch_0711.map_config
# sch_0712.map_config   need to remove last 5 empty phase group
# sch_chip2_1_debug.map_config
# IR_32core_x.xmk   clock:770000(由于时钟太大还未执行)
# IR_2group_x.xmk   前10个phase
# test_group5_case_test4_OK.py
# test_group5_case_4.py
# test_My00002.py
# test_My00003.py
# test_My00004.py
# test_ws_3chip_multicast.py
# test_wang2.py
# ir_chiptrans_receive_no_mulcast.mapping
# ir_chiptrans_receive_mul.mapping  需要整理IR

# IR_L1L2.xmk

# ir_inchiptest.mapping
ir_name = "sch_test_0807_1904.map_config"
tb_name = ir_name


def test_mapping():
    map_config = None
    with open("temp/mapping_ir/"+ir_name, "rb") as f:
        map_config = pickle.load(f)
        # print(map_config)

    # for y in range(0, 2):
    #     for x in range(14, 16):
    #         map_config[0][2].pop(((0, 0), (x, y)), None)

    # map_config[0][0].pop(((0, 0), (0, 6)), None)
    # map_config[0][0].pop(((0, 0), (0, 8)), None)
    # map_config[0][1].pop(((0, 0), (0, 2)), None)
    # map_config[0][1].pop(((0, 0), (0, 4)), None)

    # for step_id, group_id, config in MapConfig(map_config):
    #     phase_num = None
    #     if isinstance(step_id, str):
    #         continue
    #     for id in config.core_list:
    #         config._core_cofig[id]["prims"] = config._core_cofig[id]["prims"][0:5]


    # for x in range(2, 16):
    #     map_config[0][3][((0, 0), (x, 2))]['prims'][0]['soma2']= None
    #                      pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
    # for step_id, group_id, config in MapConfig(map_config):
    #     phase_num = None
    #     if isinstance(step_id, str):
    #         continue
    #     for id in config.core_list:
    #         config._core_cofig[id]["prims"][0]["router"] = None

    # ir_chiptrans_receive_mul.mapping整理IR -----------------
    # import copy
    # map_config[((1, 0), 0)] = copy.deepcopy(map_config[0])
    # map_config[((1, 1), 0)] = copy.deepcopy(map_config[0])
    # map_config[((0, 1), 0)] = copy.deepcopy(map_config[0])
    # map_config[((0, 0), 0)] = copy.deepcopy(map_config[0])
    # map_config.pop(0)
    # map_config[((1, 0), 0)].pop(2)
    # map_config[((1, 0), 0)].pop(3)
    # map_config[((1, 0), 0)].pop(4)

    # map_config[((1, 1), 0)].pop(0)
    # map_config[((1, 1), 0)].pop(1)
    # map_config[((1, 1), 0)].pop(3)
    # map_config[((1, 1), 0)].pop(4)

    # map_config[((0, 1), 0)].pop(0)
    # map_config[((0, 1), 0)].pop(1)
    # map_config[((0, 1), 0)].pop(2)
    # map_config[((0, 1), 0)].pop(4)

    # map_config[((0, 0), 0)].pop(0)
    # map_config[((0, 0), 0)].pop(1)
    # map_config[((0, 0), 0)].pop(2)
    # map_config[((0, 0), 0)].pop(3)
    # ir_chiptrans_receive_mul.mapping整理IR -----------------
    # map_config['sim_clock']=10000
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
