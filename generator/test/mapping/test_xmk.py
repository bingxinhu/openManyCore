import pickle
import os
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
import json
import numpy as np


def test_xmk():
    map_config = None

    with open("IR_phase9_10v4_6core.xmk", "rb") as f:
        # with open("IR_L1_9core.xmk", "rb") as f:
        map_config = pickle.load(f)
        print(map_config)

    for step_id, group_id, config in MapConfig(map_config):
        phase_num = None
        for id in config.core_list:
            config._core_cofig[id]["prims"] = config._core_cofig[id]["prims"][0:1]

    test_phase = []
    for step_id, group_id, config in MapConfig(map_config):
        phase_num = None
        for id in config.core_list:
            phase_num = len(config.axon_list(id))
            if phase_num is not None:
                assert phase_num == len(config.axon_list(id))
        for i in range(phase_num):
            test_phase.append((group_id, i+1))  # 确保每个group的所有core执行相同的phase数

    test_config = {
        'tb_name': "IR_phase9_10v4_6core.xmk",
        'test_mode': TestMode.MEMORY_STATE,
        'test_group_phase': test_phase
    }
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


test_xmk()
