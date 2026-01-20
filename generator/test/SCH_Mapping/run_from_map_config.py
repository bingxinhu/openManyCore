import pickle
from generator.test_engine import TestMode, TestEngine

file_name = 'sch_0717_phase0_phase4_2104_0819_changed.map_config'

with open(file_name, 'rb') as f:
    map_config = pickle.load(f)

with open('sch_0827_phaseAll.map_config', 'rb') as f1:
    map_config2 = pickle.load(f1)

group_idx_list = range(18)

case_file_name = 'sch_0718_phaseAll_0847.'


for core_x in range(8):
    for core_y in range(2):
        for i in range(3):
            map_config[((1, 2), 0)][2][((1, 2), (core_x, core_y))]['prims'][2 + i] = map_config2[((1, 2), 0)][1][((1, 2), (core_x, core_y))]['prims'][5 + i]

import pickle
with open('sch_0717_phase0_phase4_2104_changed_0828.map_config', 'wb') as f:
    pickle.dump(map_config, f)


