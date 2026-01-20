import pickle
from generator.test.SCH_Mapping.changeIR import changeIR

map_config = {
    'sim_clock': 100_000,
    ((1, 0), 0):{
        0:{
            'clock': 30_000,
            'mode': 1,
        }
    },
    ((2, 0), 0):{
        0:{
            'clock': 30_000,
            'mode': 1,
        }
    }
}
file_name = [
    'xmk_0817.map_config',
    'sch_0817.map_config',
]


with open(file_name[0], 'rb') as f:
       map_config1 = pickle.load(f)

chip = (1, 0)
map_config[((1, 0), 0)][0][(chip, (15, 0))] = {
    'prims': []
}
map_config[((1, 0), 0)][0][(chip, (15, 0))]['prims'] = [map_config1[((1, 0), 0)][2][(chip, (15, 0))]['prims'][0]]

with open(file_name[1], 'rb') as f:
    map_config2 = pickle.load(f)

chip = (2, 0)
map_config[((2, 0), 0)] = map_config2[((2, 0), 0)]

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