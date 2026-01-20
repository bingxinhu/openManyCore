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


def prim_OB11_router_Rx(from_dx, from_dy, from_dx1, from_dy1, phase, sj):
    print("prim_OB11_router_Rx", from_dx, from_dy, phase)
    Router_Prim_2 = Prim_09_Router()
    Router_Prim_2.Rhead_mode = 1
    Router_Prim_2.CXY = 0b00
    Router_Prim_2.Send_en = 0
    Router_Prim_2.Receive_en = 1
    Router_Prim_2.Addr_Dout_base = 0x400
    Router_Prim_2.Dout_Mem_sel = 0
    Router_Prim_2.Addr_Dout_length = 63
    Router_Prim_2.Send_number = 127
    Router_Prim_2.Addr_Rhead_base = 0x700
    Router_Prim_2.Addr_Rhead_length = 0
    Router_Prim_2.Addr_Din_base = 0x350
    Router_Prim_2.Addr_Din_length = 1024
    Router_Prim_2.Receive_number = 1
    Router_Prim_2.Nx = 0
    Router_Prim_2.Ny = 0
    Router_Prim_2.Send_PI_en = 0
    Router_Prim_2.Back_sign_en = 0
    Router_Prim_2.Send_PI_num = 0
    Router_Prim_2.Receive_sign_num = 0
    Router_Prim_2.Send_PI_addr_base = 0
    Router_Prim_2.Relay_number = 0
    Router_Prim_2.Q = 0
    Router_Prim_2.Receive_sign_en = 0
    Router_Prim_2.T_mode = 1
    Router_Prim_2.Soma_in_en = 0

    Router_Prim_2.Receive_PI_addr_base = 0x7a0 >> 2

    Router_Prim_2.send_destin_core_grp = []
    Router_Prim_2.recv_source_core_grp = [
        {"core_id": [((0, 0), (from_dx, from_dy)), ((0, 0), (from_dx1, from_dy1))], "data_num": 112, "T_mode": 1,
         "Rhead_num": 1, 'sync_en': 1,
         'sync_phase_num': phase}]
    Router_Prim_2.instant_prim_request = []
    Router_Prim_2.instant_request_back = []

    return Router_Prim_2


def prim_Grp8_router_Tx(dx, dy, to_dx, to_dy, phase, sj):
    Router_Prim_1 = Prim_09_Router()
    Router_Prim_1.Rhead_mode = 1
    Router_Prim_1.CXY = 0b00
    Router_Prim_1.Send_en = 1
    Router_Prim_1.Receive_en = 0
    Router_Prim_1.Addr_Dout_base = 0x350 + phase*0xE0
    Router_Prim_1.Dout_Mem_sel = 0
    Router_Prim_1.Addr_Dout_length = 55
    Router_Prim_1.Send_number = 111
    Router_Prim_1.Addr_Rhead_base = 0x300
    Router_Prim_1.Addr_Rhead_length = 6
    Router_Prim_1.Addr_Din_base = 0x800
    Router_Prim_1.Addr_Din_length = 0
    Router_Prim_1.Receive_number = 0
    Router_Prim_1.Nx = 0
    Router_Prim_1.Ny = 0
    Router_Prim_1.Send_PI_en = 0
    Router_Prim_1.Back_sign_en = 0
    Router_Prim_1.Send_PI_num = 0
    Router_Prim_1.Receive_sign_num = 0
    Router_Prim_1.Send_PI_addr_base = 0x780 >> 2  # 16B寻址
    Router_Prim_1.Relay_number = 0
    Router_Prim_1.Q = 0  # 即时原语多播
    Router_Prim_1.Receive_sign_en = 0
    Router_Prim_1.T_mode = 1
    Router_Prim_1.Soma_in_en = 0
    print("prim_Grp8_router_Tx", dx, dy, to_dx, to_dy, sj, phase, Router_Prim_1.Addr_Dout_base)
    const = 0xaa000000

    prim0_in = []
    for i in range((Router_Prim_1.Addr_Dout_length + 1) * 4):
        prim0_in.append([const + i + phase])

    Router_Prim_1.memory_blocks = [
        {'name': 'core0_0memInit',
         'start': Router_Prim_1.Addr_Dout_base + 0x8000,
         'data': prim0_in,
         'mode': 0},
    ]

    Router_Prim_1.Receive_PI_addr_base = 0x7a0 >> 2
    #  chip 1,1  1,9 -> 0,1  13,9
    Router_Prim_1.send_destin_core_grp = [
        {"core_id": [((0, 0), (to_dx, to_dy))], "data_num": 112, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1,
         'sync_phase_num': phase}]
    Router_Prim_1.recv_source_core_grp = []
    Router_Prim_1.instant_prim_request = []
    Router_Prim_1.instant_request_back = []

    if phase == 1 and sj==0:
        Router_Prim_1.Addr_Rhead_base = 0x300
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
    elif phase == 2 and sj==0:
        Router_Prim_1.Addr_Rhead_base = 0x304
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=112,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
    elif phase == 3 and sj==0:
        Router_Prim_1.Addr_Rhead_base = 0x308
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=112*2,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
    elif phase == 1 and sj==1:
        Router_Prim_1.Addr_Rhead_base = 0x30c
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=336+112*0,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
    elif phase == 2 and sj==1:
        Router_Prim_1.Addr_Rhead_base = 0x310
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=336+112*1,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
    elif phase == 3 and sj==1:
        Router_Prim_1.Addr_Rhead_base = 0x314
        Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=336+112*2,  # (1,9)
                           pack_per_Rhead=111, A_offset=0, Const=0, EN=1)


    return Router_Prim_1


prim_router_tx_0_0 = prim_Grp8_router_Tx(0, 2, 0, 9, 1, 0)  # 0,7 -> 0,9
prim_router_tx_0_1 = prim_Grp8_router_Tx(0, 2, 1, 9, 1, 0)  # 0,7 -> 0,9
prim_router_tx_0_2 = prim_Grp8_router_Tx(1, 1, 2, 9, 1, 0)  # 0,7 -> 0,9
prim_router_tx_0_3 = prim_Grp8_router_Tx(3, 1, 3, 9, 1, 0)  # 0,8 -> 3,9

prim_router_tx_1_0 = prim_Grp8_router_Tx(0, 2, 0, 9, 2, 0)  # 0,7 -> 0,9
prim_router_tx_1_1 = prim_Grp8_router_Tx(0, 2, 1, 9, 2, 0)  # 0,7 -> 0,9
prim_router_tx_1_2 = prim_Grp8_router_Tx(1, 1, 2, 9, 2, 0)  # 0,7 -> 0,9
prim_router_tx_1_3 = prim_Grp8_router_Tx(3, 1, 3, 9, 2, 0)  # 0,8 -> 3,9

prim_router_tx_2_0 = prim_Grp8_router_Tx(0, 2, 0, 9, 3, 0)  # 0,7 -> 0,9
prim_router_tx_2_1 = prim_Grp8_router_Tx(0, 2, 1, 9, 3, 0)  # 0,7 -> 0,9
prim_router_tx_2_2 = prim_Grp8_router_Tx(1, 1, 2, 9, 3, 0)  # 0,7 -> 0,9
prim_router_tx_2_3 = prim_Grp8_router_Tx(3, 1, 3, 9, 3, 0)  # 0,8 -> 3,9

prim_router_2_tx_0_0 = prim_Grp8_router_Tx(-2, 2, 0, 9, 1, 1)  # 2,7 -> 0,9
prim_router_2_tx_0_1 = prim_Grp8_router_Tx(-2, 2, 1, 9, 1, 1)  # 3,7 -> 1,9
prim_router_2_tx_0_2 = prim_Grp8_router_Tx(-1, 1, 2, 9, 1, 1)  # 3,8 -> 2,9
prim_router_2_tx_0_3 = prim_Grp8_router_Tx(1, 1, 3, 9, 1, 1)  # 2,8 -> 3,9

prim_router_2_tx_1_0 = prim_Grp8_router_Tx(-2, 2, 0, 9, 2, 1)  # 2,7 -> 0,9
prim_router_2_tx_1_1 = prim_Grp8_router_Tx(-2, 2, 1, 9, 2, 1)  # 3,7 -> 1,9
prim_router_2_tx_1_2 = prim_Grp8_router_Tx(-1, 1, 2, 9, 2, 1)  # 3,8 -> 2,9
prim_router_2_tx_1_3 = prim_Grp8_router_Tx(1, 1, 3, 9, 2, 1)  # 2,8 -> 3,9

prim_router_2_tx_2_0 = prim_Grp8_router_Tx(-2, 2, 0, 9, 3, 1)  # 2,7 -> 0,9
prim_router_2_tx_2_1 = prim_Grp8_router_Tx(-2, 2, 1, 9, 3, 1)  # 3,7 -> 1,9
prim_router_2_tx_2_2 = prim_Grp8_router_Tx(-1, 1, 2, 9, 3, 1)  # 3,8 -> 2,9
prim_router_2_tx_2_3 = prim_Grp8_router_Tx(1, 1, 3, 9,  3, 1)  # 2,8 -> 3,9

prim_router_rx_0_0 = prim_OB11_router_Rx(0, 7, 2, 7, 1, 0)  # 0,7
prim_router_rx_0_1 = prim_OB11_router_Rx(1, 7, 3, 7, 1, 0)  # 1, 7
prim_router_rx_0_2 = prim_OB11_router_Rx(1, 8, 3, 8, 1, 0)  # 1, 8
prim_router_rx_0_3 = prim_OB11_router_Rx(0, 8, 2, 8, 1, 0)  # 0, 8

prim_router_rx_1_0 = prim_OB11_router_Rx(0, 7, 2, 7, 2, 0)  # 0,7
prim_router_rx_1_1 = prim_OB11_router_Rx(1, 7, 3, 7, 2, 0)  # 1, 7
prim_router_rx_1_2 = prim_OB11_router_Rx(1, 8, 3, 8, 2, 0)  # 1, 8
prim_router_rx_1_3 = prim_OB11_router_Rx(0, 8, 2, 8, 2, 0)  # 0, 8

prim_router_rx_2_0 = prim_OB11_router_Rx(0, 7, 2, 7, 3, 0)  # 0,7
prim_router_rx_2_1 = prim_OB11_router_Rx(1, 7, 3, 7, 3, 0)  # 1, 7
prim_router_rx_2_2 = prim_OB11_router_Rx(1, 8, 3, 8, 3, 0)  # 1, 8
prim_router_rx_2_3 = prim_OB11_router_Rx(0, 8, 2, 8, 3, 0)  # 0, 8

register_dict = {
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

prim_axon = Prim_41_Axon()
prim_axon.axon_delay = True
prim_axon.Addr_InA_base = 0x0000
prim_axon.Addr_InB_base = 0x0000
prim_axon.Addr_V_base = 0x0000
prim_axon.L4_num = 8
prim_axon.L5_num = 8
prim_axon.A2S2_mode = 0

# 映射策略\
map_config = {
    'sim_clock': 20000,
    0: {
        # "cycles_number": 1,
        0: {'clock': 20000,
            'mode': 1,
            ((0, 0), (0, 7)): {
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_0_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_1_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_2_0, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (1, 7)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_0_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_1_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_2_1, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (1, 8)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_0_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_1_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_2_2, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (0, 8)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_0_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_1_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_2_3, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (2, 7)): {
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_0_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_1_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_2_0, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (3, 7)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_0_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_1_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_2_1, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (3, 8)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_0_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_1_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_2_tx_2_2, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (2, 8)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_0_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_1_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_tx_2_3, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },

            },
        1: {'clock': 3000,  # 1KB  -  4KB
            'mode': 1,
            ((0, 0), (0, 9)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_0_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_1_0, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_2_0, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (1, 9)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_0_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_1_1, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_2_1, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (2, 9)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_0_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_1_2, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_2_2, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
            ((0, 0), (3, 9)): {
                # 'prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None},
                'prims': [{'axon': prim_axon, 'soma1': None, 'router': None, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_0_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_1_3, 'soma2': None},
                          {'axon': prim_axon, 'soma1': None, 'router': prim_router_rx_2_3, 'soma2': None}],
                # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                'registers': register_dict
            },
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

print("+++++++++", test_phase)

# 测试配置
# test_config = {
#     'tb_name': tb_name,
#     'test_mode': TestMode.MEMORY_STATE,
#     # 'debug_file_switch': switch.dictionary(),
#     'test_group_phase': test_phase  # (phase_group, phase_num)
# }


# # 测试配置
from generator.test_engine.test_config import HardwareDebugFileSwitch

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
