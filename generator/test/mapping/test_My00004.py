from generator.test_engine.test_config import HardwareDebugFileSwitch
import pytest
import numpy as np
import os

from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router, Prim_06_move_merge
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig

SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
tb_name = "I11003"
np.random.seed(sum(ord(c) for c in tb_name))


Router_Prim_1 = Prim_09_Router()
Router_Prim_1.Rhead_mode = 1
Router_Prim_1.CXY = 0b00
Router_Prim_1.Send_en = 1
Router_Prim_1.Receive_en = 0
Router_Prim_1.Addr_Dout_base = 0x400
Router_Prim_1.Dout_Mem_sel = 0
Router_Prim_1.Addr_Dout_length = 111
Router_Prim_1.Send_number = 223
Router_Prim_1.Addr_Rhead_base = 0x300
Router_Prim_1.Addr_Rhead_length = 0
Router_Prim_1.Addr_Din_base = 0x800
Router_Prim_1.Addr_Din_length = 0
Router_Prim_1.Receive_number = 0
Router_Prim_1.Nx = 0
Router_Prim_1.Ny = 0
Router_Prim_1.Send_PI_en = 1
Router_Prim_1.Back_sign_en = 0
Router_Prim_1.Send_PI_num = 0
Router_Prim_1.Receive_sign_num = 0
Router_Prim_1.Send_PI_addr_base = 0x780 >> 2  # 16B寻址
Router_Prim_1.Relay_number = 0
Router_Prim_1.Q = 0
Router_Prim_1.Receive_sign_en = 1
Router_Prim_1.T_mode = 1
Router_Prim_1.Soma_in_en = 0

const = 5
prim0_in = []
for i in range((Router_Prim_1.Addr_Dout_length + 1) * 2):
    prim0_in.append([const])
const = 8
for i in range((Router_Prim_1.Addr_Dout_length + 1) * 2):
    prim0_in.append([const])
Router_Prim_1.memory_blocks = [
    {'name': 'core0_0memInit',
     'start': Router_Prim_1.Addr_Dout_base + 0x8000,
     'data': prim0_in,
     'mode': 0},
]

Router_Prim_1.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim_1.send_destin_core_grp = [
    {"core_id": ((0, 0), (1, 0)), "data_num": 112,
     "T_mode": 1, "Rhead_num": 1},
    {"core_id": ((0, 0), (1, 1)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim_1.recv_source_core_grp = []
Router_Prim_1.instant_prim_request = [(((0, 0), (1, 0)), 0)]  # 0？？
Router_Prim_1.instant_request_back = []

Router_Prim_1.add_instant_pi(
    PI_addr_offset=0, A_valid=0, S1_valid=0, R_valid=1, S2_valid=0, X=1, Y=0, Q=1)  # Q 多播

Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                       pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=1, A=0,
                       pack_per_Rhead=111, A_offset=0, Const=0, EN=1)

Router_Prim_11 = Prim_09_Router()
Router_Prim_11.Rhead_mode = 1
Router_Prim_11.CXY = 0b00
Router_Prim_11.Send_en = 1
Router_Prim_11.Receive_en = 0
Router_Prim_11.Addr_Dout_base = 0x400
Router_Prim_11.Dout_Mem_sel = 0
Router_Prim_11.Addr_Dout_length = 111
Router_Prim_11.Send_number = 223
Router_Prim_11.Addr_Rhead_base = 0x300
Router_Prim_11.Addr_Rhead_length = 0
Router_Prim_11.Addr_Din_base = 0x800
Router_Prim_11.Addr_Din_length = 0
Router_Prim_11.Receive_number = 0
Router_Prim_11.Nx = 0
Router_Prim_11.Ny = 0
Router_Prim_11.Send_PI_en = 0
Router_Prim_11.Back_sign_en = 0
Router_Prim_11.Send_PI_num = 0
Router_Prim_11.Receive_sign_num = 0
Router_Prim_11.Send_PI_addr_base = 0x780 >> 2  # 16B寻址
Router_Prim_11.Relay_number = 0
Router_Prim_11.Q = 0
Router_Prim_11.Receive_sign_en = 1
Router_Prim_11.T_mode = 1
Router_Prim_11.Soma_in_en = 0

const = 3
prim0_in = []
for i in range((Router_Prim_11.Addr_Dout_length + 1) * 2):
    prim0_in.append([const])
const = 7
for i in range((Router_Prim_11.Addr_Dout_length + 1) * 2):
    prim0_in.append([const])
Router_Prim_11.memory_blocks = [
    {'name': 'core0_0memInit',
     'start': Router_Prim_11.Addr_Dout_base + 0x8000,
     'data': prim0_in,
     'mode': 0},
]

Router_Prim_11.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim_11.send_destin_core_grp = [
    {"core_id": ((0, 0), (1, 2)), "data_num": 112,
     "T_mode": 1, "Rhead_num": 1},
    {"core_id": ((0, 0), (1, 3)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim_11.recv_source_core_grp = []
Router_Prim_11.instant_prim_request = []  # 即时原语触发
Router_Prim_11.instant_request_back = []

Router_Prim_11.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=1, A=0,
                        pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
Router_Prim_11.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=2, A=0,  # A 是指目的地的地址！
                        pack_per_Rhead=111, A_offset=0, Const=0, EN=1)

# 需要初始化数据


Router_Prim3 = Prim_09_Router()
Router_Prim3.Rhead_mode = 1
Router_Prim3.CXY = 0b00
Router_Prim3.Send_en = 0
Router_Prim3.Receive_en = 1
Router_Prim3.Addr_Dout_base = 0x400
Router_Prim3.Dout_Mem_sel = 0
Router_Prim3.Addr_Dout_length = 63
Router_Prim3.Send_number = 0
Router_Prim3.Addr_Rhead_base = 0x300
Router_Prim3.Addr_Rhead_length = 0
Router_Prim3.Addr_Din_base = 0x600
Router_Prim3.Addr_Din_length = 111
Router_Prim3.Receive_number = 0
Router_Prim3.Nx = 0
Router_Prim3.Ny = 0
Router_Prim3.Send_PI_en = 0
Router_Prim3.Back_sign_en = 1
Router_Prim3.Send_PI_num = 0
Router_Prim3.Receive_sign_num = 0
Router_Prim3.Send_PI_addr_base = 0
Router_Prim3.Relay_number = 0
Router_Prim3.Q = 1  # 多播 ！
Router_Prim3.Receive_sign_en = 0
Router_Prim3.T_mode = 1
Router_Prim3.Soma_in_en = 0
Router_Prim3.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim3.send_destin_core_grp = []
Router_Prim3.recv_source_core_grp = [
    {"core_id": ((0, 0), (0, 0)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim3.instant_prim_request = []
Router_Prim3.instant_request_back = [((0, 0), (0, 0)), ((0, 0), (0, 1))]

Router_Prim31 = Prim_09_Router()
Router_Prim31.Rhead_mode = 1
Router_Prim31.CXY = 0b00
Router_Prim31.Send_en = 0
Router_Prim31.Receive_en = 1
Router_Prim31.Addr_Dout_base = 0x400
Router_Prim31.Dout_Mem_sel = 0
Router_Prim31.Addr_Dout_length = 63
Router_Prim31.Send_number = 0
Router_Prim31.Addr_Rhead_base = 0x300
Router_Prim31.Addr_Rhead_length = 0
Router_Prim31.Addr_Din_base = 0x600
Router_Prim31.Addr_Din_length = 111
Router_Prim31.Receive_number = 0
Router_Prim31.Nx = 0
Router_Prim31.Ny = 0
Router_Prim31.Send_PI_en = 0
Router_Prim31.Back_sign_en = 0
Router_Prim31.Send_PI_num = 0
Router_Prim31.Receive_sign_num = 0
Router_Prim31.Send_PI_addr_base = 0
Router_Prim31.Relay_number = 0
Router_Prim31.Q = 0
Router_Prim31.Receive_sign_en = 0
Router_Prim31.T_mode = 1
Router_Prim31.Soma_in_en = 0
Router_Prim31.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim31.send_destin_core_grp = []
Router_Prim31.recv_source_core_grp = [
    {"core_id": ((0, 0), (0, 0)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim31.instant_prim_request = []
Router_Prim31.instant_request_back = []


Router_Prim = Prim_09_Router()
Router_Prim.Rhead_mode = 1
Router_Prim.CXY = 0b00
Router_Prim.Send_en = 0
Router_Prim.Receive_en = 1
Router_Prim.Addr_Dout_base = 0x400
Router_Prim.Dout_Mem_sel = 0
Router_Prim.Addr_Dout_length = 63
Router_Prim.Send_number = 0
Router_Prim.Addr_Rhead_base = 0x300
Router_Prim.Addr_Rhead_length = 0
Router_Prim.Addr_Din_base = 0x600
Router_Prim.Addr_Din_length = 111
Router_Prim.Receive_number = 0
Router_Prim.Nx = 0
Router_Prim.Ny = 0
Router_Prim.Send_PI_en = 0
Router_Prim.Back_sign_en = 0
Router_Prim.Send_PI_num = 0
Router_Prim.Receive_sign_num = 0
Router_Prim.Send_PI_addr_base = 0
Router_Prim.Relay_number = 0
Router_Prim.Q = 0
Router_Prim.Receive_sign_en = 0
Router_Prim.T_mode = 1
Router_Prim.Soma_in_en = 0
Router_Prim.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim.send_destin_core_grp = []
Router_Prim.recv_source_core_grp = [
    {"core_id": ((0, 0), (0, 1)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim.instant_prim_request = []
Router_Prim.instant_request_back = []

Router_Prim1 = Prim_09_Router()
Router_Prim1.Rhead_mode = 1
Router_Prim1.CXY = 0b00
Router_Prim1.Send_en = 0
Router_Prim1.Receive_en = 1
Router_Prim1.Addr_Dout_base = 0x400
Router_Prim1.Dout_Mem_sel = 0
Router_Prim1.Addr_Dout_length = 63
Router_Prim1.Send_number = 0
Router_Prim1.Addr_Rhead_base = 0x300
Router_Prim1.Addr_Rhead_length = 0
Router_Prim1.Addr_Din_base = 0x600
Router_Prim1.Addr_Din_length = 111
Router_Prim1.Receive_number = 0
Router_Prim1.Nx = 0
Router_Prim1.Ny = 0
Router_Prim1.Send_PI_en = 0
Router_Prim1.Back_sign_en = 0
Router_Prim1.Send_PI_num = 0
Router_Prim1.Receive_sign_num = 0
Router_Prim1.Send_PI_addr_base = 0
Router_Prim1.Relay_number = 0
Router_Prim1.Q = 0
Router_Prim1.Receive_sign_en = 0
Router_Prim1.T_mode = 1
Router_Prim1.Soma_in_en = 0
Router_Prim1.Receive_PI_addr_base = 0x7a0 >> 2

Router_Prim1.send_destin_core_grp = []
Router_Prim1.recv_source_core_grp = [
    {"core_id": ((0, 0), (0, 1)), "data_num": 112, "T_mode": 1, "Rhead_num": 1}]
Router_Prim1.instant_prim_request = []
Router_Prim1.instant_request_back = []


# 0x06 move
prim_soma1 = Prim_06_move_merge()
prim_soma1.length_in = 128
prim_soma1.length_ciso = 0
prim_soma1.num_in = 14
prim_soma1.num_ciso = 0
prim_soma1.length_out = 128
prim_soma1.num_out = 14
prim_soma1.type_in = 1
prim_soma1.type_out = 1
prim_soma1.in_cut_start = 0
prim_soma1.Reset_Addr_in = 1
prim_soma1.Reset_Addr_out = 1
prim_soma1.Reset_Addr_ciso = 1
prim_soma1.Row_ck_on = 0
prim_soma1.Addr_Start_in = 0x4000  # 0x4380   # 10E00
prim_soma1.Addr_Start_ciso = 0x0000  # 1E6A0
prim_soma1.Addr_Start_out = 0x8460  # 30000
prim_soma1.in_row_max = 0
prim_soma1.mem_sel = 0
prim_in = prim_soma1.init_data()
prim_soma1.memory_blocks = [
    {'name': 'prim_soma1',
        'start': prim_soma1.Addr_Start_in,
        # 'length': 672,   #   流水（不一致的时候）的时候需要声明
        'data':  prim_in[0],
        'mode': 0,
        'initialize': True},
    # {'name': 'Sending_top_buff',
    #     'start': prim_soma1.Addr_Start_out,
    #     'length': 672,
    #     'data': prim_in[1],
    #     'mode': 0},
]


# 映射策略\
map_config = {
    'sim_clock': 20000,
    0: {
        # 'clock0_in_step':
        # 'clock1_in_step':
        # 'cycles_num': math.floor(map_config.get("sim_clk") / map_config[0].get("clock0_in_step"))
        0: {'clock': 15000,
            'mode': 1,
            ((0, 0), (0, 0)): {
                'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_1, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0,
                    "PI_CXY": 0,
                    "PI_Nx": 0,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 1,  # 即时原语应答的多播
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 1,
                    "instant_PI_en": 0,
                    "fixed_instant_PI": 0,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0}
            },
            ((0, 0), (0, 1)): {
                'prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_11, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0,
                    "PI_CXY": 0,
                    "PI_Nx": 0,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 0,
                    "fixed_instant_PI": 0,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0}
            }
            },
        1: {'clock': 20000,
            'mode': 1,
            ((0, 0), (1, 0)): {
                'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'instant_prims': [{'axon': None, 'soma1': None, 'router': Router_Prim3, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
                    "PI_CXY": 1,
                    "PI_Nx": 0,
                    "PI_Ny": 1,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0  # 第1个phase结束后执行即时原语
                }
            },
            ((0, 0), (1, 1)): {
                'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'instant_prims': [{'axon': None, 'soma1': None, 'router': Router_Prim31, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
                    "PI_CXY": 1,
                    "PI_Nx": 0,
                    "PI_Ny": 1,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0  # 第1个phase结束后执行即时原语
                }
            },
            ((0, 0), (1, 2)): {
                'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'instant_prims': [{'axon': None, 'soma1': None, 'router': Router_Prim, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
                    "PI_CXY": 1,
                    "PI_Nx": 0,
                    "PI_Ny": 1,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0  # 第1个phase结束后执行即时原语
                }
            },
            ((0, 0), (1, 3)): {
                'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'instant_prims': [{'axon': None, 'soma1': None, 'router': Router_Prim1, 'soma2': None}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
                    "PI_CXY": 0,
                    "PI_Nx": 0,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0  # 第1个phase结束后执行即时原语
                }
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
        test_phase.append((group_id, i + 1))  # 确保每个group的所有core执行相同的phase数

print("+++++", test_phase)

# 测试配置
test_config = {
    'tb_name': 'I11003',
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
