import pickle

file_name = [
    'mapping_ir_merged.mapping',
]

with open(file_name[0], 'rb') as f:
    map_config2 = pickle.load(f)
# map_config2 = changeIR(map_config2, chip_x=3, chip_y=3, group_idx_list=[0, 1])


from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch

test_config = {
        'tb_name': 'M66666',
        'test_mode': TestMode.MEMORY_STATE,
        # 'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'debug_file_switch': HardwareDebugFileSwitch().multi_chip.close_all.open_debug_message.dict,
        'test_group_phase': [(0, 1)]
    }

tester = TestEngine(map_config2, test_config)
assert tester.run_test()