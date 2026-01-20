import pytest
import numpy as np
import os
import copy

from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router, Prim_06_move_merge, Prim_06_move_split, Prim_02_Axon
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig

SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
np.random.seed(sum(ord(c) for c in tb_name))


def prim_delay():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x4000
    prim_axon.Addr_InB_base = 0x2000
    prim_axon.Addr_V_base = 0x1000
    prim_axon.L4_num = 8
    prim_axon.L5_num = 8
    return prim_axon


def prim_tx_core0():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x4000
    prim_axon.Addr_InB_base = 0x2000
    prim_axon.Addr_V_base = 0x1000
    prim_axon.L4_num = 8
    prim_axon.L5_num = 8

    # move -> mem3 发送
    prim_soma1 = Prim_06_move_merge()
    prim_soma1.length_in = 256
    prim_soma1.length_ciso = 0
    prim_soma1.num_in = 28
    prim_soma1.num_ciso = 0
    prim_soma1.length_out = 256
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

    memblock_num = "prim_buf_core_tx_phase"
    prim_in = prim_soma1.init_data()
    data = []
    for i in range(len(prim_in[0])):
        data.append([3, 3, 3, 3])
    prim_soma1.memory_blocks = [
        {'name': memblock_num,
         'start': prim_soma1.Addr_Start_in,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': data,
         'mode': 0,
         'initialize': True}
    ]

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 1
    prim_router.Receive_en = 0
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 447
    prim_router.Send_number = 895
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x800
    prim_router.Addr_Din_length = 255
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
        {"core_id": [((0, 0), (0, 1))], "data_num": 896, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1, 'sync_phase_num': 0}]  # ,'sync_en':1,'sync_phase_num':0
    prim_router.recv_source_core_grp = []
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0,
                         pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
    # prim_soma1=None
    return prim_axon, prim_soma1, prim_router


def prim_rx_core03():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x4000
    prim_axon.Addr_InB_base = 0x2000
    prim_axon.Addr_V_base = 0x1000
    prim_axon.L4_num = 8
    prim_axon.L5_num = 8

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 0
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 447
    prim_router.Send_number = 895
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 895
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
        {"core_id": [((0, 0), (0, 0))], "data_num": 896, "T_mode": 1, "Rhead_num": 1, 'sync_en': 1, 'sync_phase_num': 0}]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0,
    #                      pack_per_Rhead=895, A_offset=0, Const=0, EN=1)

    return prim_axon, prim_router


def prim_core00_list_config(i, j):
    prims_list = []
    prims_dict = {}

    prims_dict = copy.deepcopy(prims_dict)
    prim_axon, prim_soma1, prim_router = prim_tx_core0()
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = prim_soma1
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = None
    prims_list.append(prims_dict)

    return prims_list


def prim_core01_list_config(i, j):
    prims_list = []
    prims_dict = {}

    prims_dict = copy.deepcopy(prims_dict)
    prim_axon, prim_router = prim_rx_core03()
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = None
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = None
    prims_list.append(prims_dict)

    return prims_list


def prim_core02_list_config(i, j):
    prims_list = []
    prims_dict = {}

    print("C02", i, j)

    prims_dict = copy.deepcopy(prims_dict)

    prims_dict = copy.deepcopy(prims_dict)
    prim_axon = prim_delay()
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = None
    prims_dict['router'] = None
    prims_dict['soma2'] = None
    prims_list.append(prims_dict)

    return prims_list


def registers_dict(i, j):
    registers_dict = {"Receive_PI_addr_base": 0,
                      "PI_CXY": 0,
                      "PI_Nx": 0,
                      "PI_Ny": 0,
                      "PI_sign_CXY": 0,  # 即时原语应答的多播
                      "PI_sign_Nx": 0,
                      "PI_sign_Ny": 0,
                      "instant_PI_en": 0,
                      "fixed_instant_PI": 0,
                      "instant_PI_number": 0,
                      "PI_loop_en": 0,
                      "start_instant_PI_num": 0}

    if i == 0 and j == 0:
        pass
    elif i == 0 and j == 1:
        pass
    return registers_dict


def instant_prims_list(i, j):
    instant_prims_list = []

    if i == 0 and j == 0:
        pass
    elif i == 0 and j == 1:
        pass

    return instant_prims_list


def prims_list(i, j):
    prims_list = []
    prims_list.clear()
    print("prims_list", i, j)
    if i == 0 and j == 0:
        prims_list = prim_core00_list_config(i=0, j=0)
    elif i == 0 and j == 1:
        prims_list = prim_core01_list_config(i=0, j=1)
    elif i == 0 and j == 2:
        prims_list = prim_core02_list_config(i=0, j=2)

    print(prims_list, type(prims_list))

    return prims_list


def core_config(i, j):
    print("core_config", i, j)
    core_config = {}
    core_config['prims'] = prims_list(i=i, j=j)
    core_config['instant_prims'] = instant_prims_list(i=i, j=j)
    core_config['registers'] = registers_dict(i=i, j=j)
    return core_config


def group0_config():
    group0_config = {}
    group0_config['clock'] = 8000
    group0_config['mode'] = 1
    # 第1个列条
    group0_config[((0, 0), (0, 0))] = core_config(i=0, j=0)

    return group0_config


def group1_config():
    group1_config = {}
    group1_config['clock'] = 8000
    group1_config['mode'] = 1
    # 第1个列条
    group1_config[((0, 0), (0, 1))] = core_config(i=0, j=1)

    return group1_config


def group2_config():
    group2_config = {}
    group2_config['clock'] = 20000
    group2_config['mode'] = 1
    # 第1个列条
    group2_config[((0, 0), (0, 2))] = core_config(i=0, j=2)

    return group2_config


def group3_config():
    group3_config = {}
    group3_config['clock'] = 20000
    group3_config['mode'] = 1
    # 第1个列条
    group3_config[((0, 0), (0, 3))] = core_config(i=0, j=3)

    return group3_config


# 映射策略\
map_config = {
    'sim_clock': 20000,
    0: {
        0: group0_config(),  # tx
        1: group1_config(),  # work sector1
        2: group2_config(),  # work sector2
        # 3: group3_config()  # rx
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
from generator.test_engine.test_config import HardwareDebugFileSwitch
test_config = {
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    # 'debug_file_switch':HardwareDebugFileSwitch().close_all().dict,
    'test_group_phase': test_phase  # (phase_group, phase_num)
}


# 开始测试


def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
