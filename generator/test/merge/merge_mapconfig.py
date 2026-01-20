import pickle
from generator.test.SCH_Mapping.changeIR import changeIR

map_config = {
    'sim_clock': 100_000,
    ((1, 1), 0):{
        0:{
            'clock': 30_000,
            'mode': 1,
        }
    },
    ((1, 2), 0):{
        0:{
            'clock': 30_000,
            'mode': 1,
        }
    }
}
file_name = [
    'OB11_ir_0815.map',
    'sch_0809_1406.map_config',
]


with open(file_name[0], 'rb') as f:
       map_config1 = pickle.load(f)

chip = (1, 1)
for (core_x, core_y) in [(8, 9), (9, 9), (10, 9), (11, 9)]:
    map_config[((1, 1), 0)][0][(chip, (core_x, core_y))] = {}
    # map_config[((1, 1), 0)][0][(chip, (core_x, core_y))]['prims'] = []
    map_config[((1, 1), 0)][0][(chip, (core_x, core_y))]['prims'] = map_config1[((1, 1), 0)][0][(chip, (core_x, core_y))]['prims']


with open(file_name[1], 'rb') as f:
    map_config2 = pickle.load(f)
# map_config2 = changeIR(map_config2, chip_x=3, chip_y=3, group_idx_list=[0])

chip = (1, 2)
for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
    map_config[((1, 2), 0)][0][(chip, (core_x, core_y))] = {}
    # map_config[((1, 2), 1)][0][(chip, (core_x, core_y))]['prims'] = []
    map_config[((1, 2), 0)][0][(chip, (core_x, core_y))]['prims'] = map_config2[((1, 2), 0)][0][(chip, (core_x, core_y))]['prims']

from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch

test_config = {
        'tb_name': 'M99999',
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        # 'debug_file_switch': HardwareDebugFileSwitch().open_debug_message.dict,
        'test_group_phase': [(0, 1)]
    }

tester = TestEngine(map_config, test_config)
assert tester.run_test()