from generator.test_engine.test_config import HardwareDebugFileSwitch
import pytest
import numpy as np
import os

from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]

np.random.seed(sum(ord(c) for c in tb_name))


Router_Prim_1 = Prim_09_Router()
Router_Prim_1.Rhead_mode = 1
Router_Prim_1.CXY = 0b00
Router_Prim_1.Send_en = 1
Router_Prim_1.Receive_en = 1
Router_Prim_1.Addr_Dout_base = 0x400
Router_Prim_1.Dout_Mem_sel = 0
Router_Prim_1.Addr_Dout_length = 3
Router_Prim_1.Send_number = 7
Router_Prim_1.Addr_Rhead_base = 0x300
Router_Prim_1.Addr_Rhead_length = 0
Router_Prim_1.Addr_Din_base = 0x800
Router_Prim_1.Addr_Din_length = 31
Router_Prim_1.Receive_number = 0
Router_Prim_1.Nx = 0
Router_Prim_1.Ny = 0
Router_Prim_1.Send_PI_en = 0
Router_Prim_1.Back_sign_en = 0
Router_Prim_1.Send_PI_num = 0
Router_Prim_1.Receive_sign_num = 0
Router_Prim_1.Send_PI_addr_base = 0  # 16B寻址
Router_Prim_1.Relay_number = 0
Router_Prim_1.Q = 0
Router_Prim_1.Receive_sign_en = 0
Router_Prim_1.T_mode = 1
Router_Prim_1.Soma_in_en = 0


prim0_in = []
for i in range(16):
    prim0_in.append([i])

Router_Prim_1.memory_blocks = [
    {'name': 'core0_0memInit',
        'start': Router_Prim_1.Addr_Dout_base + 0x8000,
        'data': prim0_in,
        'mode': 0},
]

Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0,
                       pack_per_Rhead=7, A_offset=1, Const=0, EN=1)
# 映射策略\
map_config = {
    'sim_clock': 5000,
    0: {
        0: {'clock': 5000,
            'mode': 1,
            ((0, 0), (0, 0)): {
                'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_1, 'soma2': None}],
            }
            }
    }
}


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
# 测试配置
test_config = {
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    'debug_file_switch': HardwareDebugFileSwitch().close_debug_message.singla_chip.dict,
    'test_group_phase': test_phase  # (phase_group, phase_num)
}


# 开始测试


def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
