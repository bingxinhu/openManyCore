import pytest
import numpy as np
import os

from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
from generator.test_engine.test_config import HardwareDebugFileSwitch
SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
np.random.seed(sum(ord(c) for c in tb_name))

# phase_grp0;core(0,0) 静态原语phase1

Router_Prim_1 = Prim_09_Router()
Router_Prim_1.Rhead_mode = 1
Router_Prim_1.CXY = 0b00
Router_Prim_1.Send_en = 1
Router_Prim_1.Receive_en = 0
Router_Prim_1.Addr_Dout_base = 0x400
Router_Prim_1.Dout_Mem_sel = 0
Router_Prim_1.Addr_Dout_length = 63
Router_Prim_1.Send_number = 127
Router_Prim_1.Addr_Rhead_base = 0x700
Router_Prim_1.Addr_Rhead_length = 0
Router_Prim_1.Addr_Din_base = 0x800
Router_Prim_1.Addr_Din_length = 0
Router_Prim_1.Receive_number = 0
Router_Prim_1.Nx = 0
Router_Prim_1.Ny = 0
Router_Prim_1.Send_PI_en = 0
Router_Prim_1.Back_sign_en = 0
Router_Prim_1.Send_PI_num = 0
Router_Prim_1.Receive_sign_num = 0
Router_Prim_1.Send_PI_addr_base = 0x780 >> 2    # 16B寻址
Router_Prim_1.Relay_number = 0
Router_Prim_1.Q = 0  # 即时原语多播
Router_Prim_1.Receive_sign_en = 0
Router_Prim_1.T_mode = 1
Router_Prim_1.Soma_in_en = 0

const = 0xcafebabe
prim0_in = []
for i in range((Router_Prim_1.Addr_Dout_length+1)*4):
    prim0_in.append([const])

Router_Prim_1.memory_blocks = [
    {'name': 'core0_0memInit',
        'start': Router_Prim_1.Addr_Dout_base+0x8000,
        'data': prim0_in,
        'mode': 0},
]

Router_Prim_1.Receive_PI_addr_base = 0x7a0 >> 2
#  chip 1,1  1,9 -> 0,1  13,9
Router_Prim_1.send_destin_core_grp = [
    {"core_id": [((0, 1), (13, 9)), ((0, 2), (13, 0))], "data_num": 128, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1,
     'sync_phase_num': 0}]
Router_Prim_1.recv_source_core_grp = []
Router_Prim_1.instant_prim_request = []
Router_Prim_1.instant_request_back = []

Router_Prim_1.addRHead(S=0, T=1, P=0, Q=1, X=-4, Y=0, A=0,  # (1,9)
                       pack_per_Rhead=127, A_offset=0, Const=0, EN=1)

# 需要初始化数据

# phase_grp1;core(1,0) 静态原语phase1

Router_Prim_2 = Prim_09_Router()
Router_Prim_2.Rhead_mode = 1
Router_Prim_2.CXY = 0b01
Router_Prim_2.Send_en = 0
Router_Prim_2.Receive_en = 1
Router_Prim_2.Addr_Dout_base = 0x400
Router_Prim_2.Dout_Mem_sel = 0
Router_Prim_2.Addr_Dout_length = 63
Router_Prim_2.Send_number = 127
Router_Prim_2.Addr_Rhead_base = 0x700
Router_Prim_2.Addr_Rhead_length = 0
Router_Prim_2.Addr_Din_base = 0x800
Router_Prim_2.Addr_Din_length = 127
Router_Prim_2.Receive_number = 0
Router_Prim_2.Nx = 0
Router_Prim_2.Ny = 1
Router_Prim_2.Send_PI_en = 0
Router_Prim_2.Back_sign_en = 0
Router_Prim_2.Send_PI_num = 0
Router_Prim_2.Receive_sign_num = 0
Router_Prim_2.Send_PI_addr_base = 0
Router_Prim_2.Relay_number = 127
Router_Prim_2.Q = 0
Router_Prim_2.Receive_sign_en = 0
Router_Prim_2.T_mode = 1
Router_Prim_2.Soma_in_en = 0


const = 0xdeadbeef
prim1_in = []
for i in range((Router_Prim_2.Addr_Dout_length+1)*4):
    prim1_in.append([const])

Router_Prim_2.memory_blocks = [
    {'name': 'core1_0memInit',
        'start': Router_Prim_2.Addr_Dout_base+0x8000,
        'data': prim1_in,
        'mode': 0},
]

Router_Prim_2.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim_2.send_destin_core_grp = []
Router_Prim_2.recv_source_core_grp = [
    {"core_id": ((1, 1), (1, 9)), "data_num": 128, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1,
     'sync_phase_num': 0}]
Router_Prim_2.instant_prim_request = []
Router_Prim_2.instant_request_back = []


# 需要初始化数据

# phase_grp1;core(1,0) 即时原语phase1

Router_Prim_3 = Prim_09_Router()
Router_Prim_3.Rhead_mode = 1
Router_Prim_3.CXY = 0b00
Router_Prim_3.Send_en = 0
Router_Prim_3.Receive_en = 1
Router_Prim_3.Addr_Dout_base = 0x400
Router_Prim_3.Dout_Mem_sel = 0
Router_Prim_3.Addr_Dout_length = 63
Router_Prim_3.Send_number = 0
Router_Prim_3.Addr_Rhead_base = 0x700
Router_Prim_3.Addr_Rhead_length = 0
Router_Prim_3.Addr_Din_base = 0x800
Router_Prim_3.Addr_Din_length = 127
Router_Prim_3.Receive_number = 0
Router_Prim_3.Nx = 0
Router_Prim_3.Ny = 0
Router_Prim_3.Send_PI_en = 0
Router_Prim_3.Back_sign_en = 0
Router_Prim_3.Send_PI_num = 0
Router_Prim_3.Receive_sign_num = 0
Router_Prim_3.Send_PI_addr_base = 0
Router_Prim_3.Relay_number = 0
Router_Prim_3.Q = 0
Router_Prim_3.Receive_sign_en = 0
Router_Prim_3.T_mode = 0
Router_Prim_3.Soma_in_en = 0
Router_Prim_3.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim_3.send_destin_core_grp = []
Router_Prim_3.recv_source_core_grp = [
    {"core_id": ((1, 1), (1, 9)), "data_num": 128, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1,
     'sync_phase_num': 0}]
Router_Prim_3.instant_prim_request = []
Router_Prim_3.instant_request_back = []


# 映射策略\
map_config = {
    'sim_clock': 3000,
    ((1, 1), 0): {
        "cycles_number": 1,
        0: {'clock': 3000,
            'mode': 1,
            ((1, 1), (1, 9)): {
                'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_1, 'soma2': None}],
            },
            },
    },
    ((0, 1), 0): {
        1:  {'clock': 3000,  # 1KB  -  4KB
             'mode': 1,
             ((0, 1), (13, 9)): {
                 'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_2, 'soma2': None}],
            }
            },

    },
    ((0, 2), 0): {
        2:  {'clock': 3000,  # 1KB  -  4KB
             'mode': 1,
             ((0, 2), (13, 0)): {
                 'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_3, 'soma2': None}],
            }
        }
    }
}


switch = HardwareDebugFileSwitch()
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
        test_phase.append((group_id, i + 1))  # 确保每个group的所有core执行相同的phase数
# 测试配置
test_config = {
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    'debug_file_switch':HardwareDebugFileSwitch().close_all.singla_chip.dict,
    'test_group_phase': test_phase    # (phase_group, phase_num)
}

# 开始测试


def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
