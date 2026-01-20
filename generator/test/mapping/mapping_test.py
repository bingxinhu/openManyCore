import pickle
import os
from generator.test_engine import TestMode, TestEngine


# 36-170,237-252,269-316
test = [(36, 170), (237, 252), (269, 316)]

for i in test:
    for j in range(i[0], i[1]+1):
        print(j)
        tb_name = ""
        if j <= 9:
            tb_name = "M0000" + str(j)
        elif j <= 99:
            tb_name = "M000" + str(j)
        else:
            tb_name = "M00" + str(j)

        with open('temp/router_map_config/'+tb_name+'.map_config', 'rb') as f:
            map_config = pickle.load(f)

        test_phase = []
        for i in range(len(map_config[0][0][((0, 0), (0, 0))]['axon'])):
            test_phase.append((0, i + 1))
        test_config = {
            'tb_name': tb_name,
            'test_mode': TestMode.MEMORY_STATE,
            'test_group_phase': test_phase
        }

        tester = TestEngine(map_config, test_config)
        assert tester.run_test()
