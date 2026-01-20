from generator.test_engine.test_config import HardwareDebugFileSwitch
import pytest
import numpy as np
import os
from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router, Prim_06_move_merge, Prim_06_move_split, Prim_02_Axon
from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig

SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
np.random.seed(sum(ord(c) for c in tb_name))

DELAY_Num = 12


# phase_grp0;core(0,0) 静态原语phase1

def Group0_tx_core():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_Num
    prim_axon.L5_num = DELAY_Num
    prim_axon.A2S2_mode = 1

    # # move -> mem3 发送
    prim_soma1 = Prim_06_move_merge()
    prim_soma1.length_in = 128
    prim_soma1.length_ciso = 0
    prim_soma1.num_in = 28
    prim_soma1.num_ciso = 28
    prim_soma1.length_out = 128
    prim_soma1.num_out = 28
    prim_soma1.type_in = 1
    prim_soma1.type_out = 1
    prim_soma1.in_cut_start = 0
    prim_soma1.Reset_Addr_in = 1
    prim_soma1.Reset_Addr_out = 1
    prim_soma1.Reset_Addr_ciso = 1
    prim_soma1.Row_ck_on = 0
    prim_soma1.Addr_Start_in = 0x4000  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "core_tx_phase"
    prim_in = prim_soma1.init_data()
    prim_soma1.memory_blocks = [
        {'name': memblock_num,
         'start': prim_soma1.Addr_Start_in,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
         'mode': 0,
         'initialize': True}
    ]

    # for i in range(len(prim_in[0])):
    #     print("i={} : ".format(i),prim_in[0][i])

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 1
    prim_router.Receive_en = 0
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 447
    prim_router.Receive_number = 0
    prim_router.Nx = 0
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 0
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [
        {'core_id': [((0, 0), (1, 0))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': 0},
    ]
    prim_router.recv_source_core_grp = []
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                         pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # prim_soma2 = Prim_06_move_merge()
    # prim_soma2.length_in = 512
    # prim_soma2.length_out = 512
    # prim_soma2.length_ciso = 512
    # prim_soma2.num_in = 28
    # prim_soma2.num_out = 28
    # prim_soma2.num_ciso = 28
    # prim_soma2.type_in = 1
    # prim_soma2.type_out = 1
    # prim_soma2.in_cut_start = 0
    # prim_soma2.Reset_Addr_in = 1
    # prim_soma2.Reset_Addr_out = 1
    # prim_soma2.Reset_Addr_ciso = 1
    # prim_soma2.Row_ck_on = 0
    # prim_soma2.Addr_Start_in = 0x0000  # mem2
    # prim_soma2.Addr_Start_out = 0x2000  #
    # prim_soma2.Addr_Start_ciso = 0x6000  # Null
    # prim_soma2.in_row_max = 0

    return prim_axon, prim_soma1, prim_router, None


def Group1_tx_rx_core():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_Num
    prim_axon.L5_num = DELAY_Num
    prim_axon.A2S2_mode = 1

    # # move -> mem3 发送
    prim_soma1 = Prim_06_move_merge()
    prim_soma1.length_in = 128
    prim_soma1.length_ciso = 128
    prim_soma1.num_in = 28
    prim_soma1.num_ciso = 28
    prim_soma1.length_out = 128
    prim_soma1.num_out = 28
    prim_soma1.type_in = 1
    prim_soma1.type_out = 1
    prim_soma1.in_cut_start = 0
    prim_soma1.Reset_Addr_in = 1
    prim_soma1.Reset_Addr_out = 1
    prim_soma1.Reset_Addr_ciso = 1
    prim_soma1.Row_ck_on = 0
    prim_soma1.Addr_Start_in = 0x4000  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "core_tx_phase11" + str(0)
    prim_in = prim_soma1.init_data()
    prim_soma1.memory_blocks = [
        {'name': memblock_num,
         'start': prim_soma1.Addr_Start_in,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
         'mode': 0,
         'initialize': True}
    ]

    # for i in range(len(prim_in[0])):
    #     print("i={} : ".format(i),prim_in[0][i])

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 1
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 447
    prim_router.Receive_number = 0
    prim_router.Nx = 0
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 0
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [
        {'core_id': [((0, 0), (2, 0))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': 0},
    ]
    prim_router.recv_source_core_grp = [
        {'core_id': [((0, 0), (0, 0))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': 0}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                         pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 128
    prim_soma2.length_out = 128
    prim_soma2.length_ciso = 128
    prim_soma2.num_in = 28
    prim_soma2.num_out = 28
    prim_soma2.num_ciso = 28
    prim_soma2.type_in = 1
    prim_soma2.type_out = 1
    prim_soma2.in_cut_start = 0
    prim_soma2.Reset_Addr_in = 1
    prim_soma2.Reset_Addr_out = 1
    prim_soma2.Reset_Addr_ciso = 1
    prim_soma2.Row_ck_on = 0
    prim_soma2.Addr_Start_in = 0x8400  # mem2
    prim_soma2.Addr_Start_out = 0x6000  #
    prim_soma2.Addr_Start_ciso = 0x7000  # Null
    prim_soma2.in_row_max = 0

    # prim_soma2 = None
    return prim_axon, prim_soma1, prim_router, prim_soma2


def Group2_rx_core():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_Num
    prim_axon.L5_num = DELAY_Num
    prim_axon.A2S2_mode = 1

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 0
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x900
    prim_router.Addr_Din_length = 447
    prim_router.Receive_number = 0
    prim_router.Nx = 0
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 0
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 0

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = []
    prim_router.recv_source_core_grp = [
        {'core_id': [((0, 0), (1, 0))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': 0},
    ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 128
    prim_soma2.length_out = 128
    prim_soma2.length_ciso = 128
    prim_soma2.num_in = 28
    prim_soma2.num_out = 28
    prim_soma2.num_ciso = 28
    prim_soma2.type_in = 1
    prim_soma2.type_out = 1
    prim_soma2.in_cut_start = 0
    prim_soma2.Reset_Addr_in = 1
    prim_soma2.Reset_Addr_out = 1
    prim_soma2.Reset_Addr_ciso = 1
    prim_soma2.Row_ck_on = 0
    prim_soma2.Addr_Start_in = 0x8900  # mem2
    prim_soma2.Addr_Start_out = 0x6000  #
    prim_soma2.Addr_Start_ciso = 0x7000  # Null
    prim_soma2.in_row_max = 0

    # prim_soma2 = None

    return prim_axon, prim_router, prim_soma2


Grp0_axon, Grp0_soma1, Grp0_router, Grp0_soma2 = Group0_tx_core()
Grp1_axon, Grp1_soma1, Grp1_router, Grp1_soma2 = Group1_tx_rx_core()
Grp2_axon, Grp2_router, Grp2_soma2 = Group2_rx_core()
# 映射策略\
map_config = {
    'sim_clock': 20000,
    0: {
        "cycles_number": 1,
        0: {'clock': 10000,
            'mode': 1,
            ((0, 0), (0, 0)): {
                'prims': [{'axon': Grp0_axon, 'soma1': Grp0_soma1, 'router': Grp0_router, 'soma2': None}],
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
        1: {'clock': 10000,
            'mode': 1,
            ((0, 0), (1, 0)): {
                'prims': [{'axon': Grp1_axon, 'soma1': Grp1_soma1, 'router': Grp1_router, 'soma2': Grp1_soma2}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
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
        2: {'clock': 10000,
            'mode': 1,
            ((0, 0), (2, 0)): {
                'prims': [{'axon': Grp2_axon, 'soma1': None, 'router': Grp2_router, 'soma2': Grp2_soma2}],
                'registers': {
                    "Receive_PI_addr_base": 0x7a0 >> 2,
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
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    'debug_file_switch': HardwareDebugFileSwitch().close_all.singla_chip.dict,
    'test_group_phase': test_phase  # (phase_group, phase_num)
}


# 开始测试


def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
