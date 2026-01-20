from generator.test_engine.test_config import HardwareDebugFileSwitch
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

SIM_CLOCK = 30000
GRP0_CLOCK = 20000
GRP1_CLOCK = 20000
GRP2_CLOCK = 20000
GRP3_CLOCK = 20000
GRP4_CLOCK = 20000

DELAY_L4_Num = 15  #
DELAY_L5_Num = 15  #


# -------------------------------Delay 原语-------------------------------------------

def prim_delay():
    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x4000
    prim_axon.Addr_InB_base = 0x4000
    prim_axon.Addr_V_base = 0x4000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 0
    return prim_axon


# -------------------------------数据收发 原语-------------------------------------------

def prim_IB11_Tx_core(m, n, i, j, phase):
    router_list_rhead = [(-14, 0), (-12, -1), (-10, -2),
                         (-8, -3), (-6, -4), (-4, -5), (-2, -6)]
    router_list_send = [(0, 0), (2, 0), (4, 0), (6, 0),
                        (8, 0), (10, 0), (12, 0)]
    router_list_Multicast = [[(1, 0), (1, 1), (0, 1), (0, 2), (1, 2)],  # 第一列条
                             [(3, 0), (3, 1), (2, 1), (2, 2), (3, 2)],  # 第二列条
                             [(5, 0), (5, 1), (4, 1), (4, 2), (5, 2)],  # 第三列条
                             [(7, 0), (7, 1), (6, 1), (6, 2), (7, 2)],  # 第四列条
                             [(9, 0), (9, 1), (8, 1), (8, 2), (9, 2)],  # 第五列条
                             [(11, 0), (11, 1), (10, 1),
                              (10, 2), (11, 2)],  # 第六列条
                             [(13, 0), (13, 1), (12, 1),
                              (12, 2), (13, 2)],  # 第七列条
                             ]
    dx, dy = router_list_rhead[j]
    to_dx, to_dy = router_list_send[j]
    to_dx1, to_dy1 = router_list_Multicast[j][0]
    to_dx2, to_dy2 = router_list_Multicast[j][1]
    to_dx3, to_dy3 = router_list_Multicast[j][2]
    to_dx4, to_dy4 = router_list_Multicast[j][3]
    to_dx5, to_dy5 = router_list_Multicast[j][4]

    print("prim_IB11_Tx_core", dx, dy, to_dx, to_dy, to_dx1, to_dy1, to_dx2, to_dy2, to_dx3, to_dy3, to_dx4, to_dy4,
          to_dx5, to_dy5)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

    # # move -> mem3 发送
    prim_soma1 = Prim_06_move_merge()
    prim_soma1.length_in = 256
    prim_soma1.length_ciso = 256
    prim_soma1.num_in = 28
    prim_soma1.num_ciso = 28
    prim_soma1.length_out = 256
    prim_soma1.num_out = 28
    prim_soma1.type_in = 1
    prim_soma1.type_out = 1
    prim_soma1.in_cut_start = 0
    prim_soma1.Reset_Addr_in = 1
    prim_soma1.Reset_Addr_out = 1
    prim_soma1.Reset_Addr_ciso = 1
    prim_soma1.Row_ck_on = 0
    prim_soma1.Addr_Start_in = 0x4000 + 0x700 * phase  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "prim_buf_core_tx_phase" + str(phase)
    prim_in = prim_soma1.init_data()
    prim_soma1.memory_blocks = [
        {'name': memblock_num,
         'start': prim_soma1.Addr_Start_in,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
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
    prim_router.Addr_Din_length = 0
    prim_router.Receive_number = 0
    prim_router.Nx = 0
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 895
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [
        {'core_id': [((m, n), (to_dx, to_dy)), ((m, n), (to_dx1, to_dy1)), ((m, n), (to_dx2, to_dy2)),
                     ((m, n), (to_dx3, to_dy3)), ((m, n), (to_dx4, to_dy4)), ((m, n), (to_dx5, to_dy5))],
         'data_num': 896, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': phase}]
    prim_router.recv_source_core_grp = []
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=1, Q=1, X=dx, Y=dy, A=0,
                         pack_per_Rhead=895, A_offset=0, Const=0, EN=1)

    return prim_axon, prim_soma1, prim_router, None


# i==0,2,4,6,8,10,12 发送本core数据，并多播接收到的数据
def prim_grp5_rx_tx_core(m, n, i, j, phase):
    router_list_rhead = [(0, 3), (0, 3), (0, 3), (0, 3),
                         (0, 3), (0, 3), (0, 3)]
    router_list_send = [(0, 3), (2, 3), (4, 3), (6, 3),
                        (8, 3), (10, 3), (12, 3)]
    router_list_recv = [(14, 0), (14, 1), (14, 2),
                        (14, 3), (14, 4), (14, 5), (14, 6)]

    dx, dy = router_list_rhead[i // 2]
    to_dx, to_dy = router_list_send[i // 2]
    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_grp5_rx_tx_core", dx, dy, to_dx, to_dy, from_dx, from_dy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

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
    prim_soma1.Addr_Start_in = 0x4000 + 0x380 * phase  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "core_tx_phase" + str(phase)
    prim_in = prim_soma1.init_data()

    data = []
    for i in range(len(prim_in[0])):
        # type_in 类型，若int8,3,int32,....
        data.append([phase + 3, phase + 3, phase + 3, phase + 3])

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
    prim_router.CXY = 0b01
    prim_router.Send_en = 1
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 895
    prim_router.Receive_number = 0
    prim_router.Nx = 1
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 895
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [{'core_id': [((m, n), (to_dx, to_dy))], 'data_num': 448,
                                         'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': phase}]
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 896, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=1, X=dx, Y=dy, A=0,
                         pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 256
    prim_soma2.length_out = 256
    prim_soma2.length_ciso = 256
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x700 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, prim_soma1, prim_router, prim_soma2


# i！= 0,2,4,6,8,10,12     接收多播数据,并多播到shortcout组
def prim_grp5_multicast_rx_core(m, n, i, j, phase):

    Nx = Ny = 0
    Cxy = 0

    router_list_recv = [(14, 0), (14, 1), (14, 2),
                        (14, 3), (14, 4), (14, 5), (14, 6)]
    router_list_Nxy = [(0, 1), (-1, 0), (0, 1)]

    if i % 2 == 1 and j == 0:
        Nx, Ny = router_list_Nxy[0]
    elif i % 2 == 1 and j == 1:
        Nx, Ny = router_list_Nxy[1]
    elif i % 2 == 0 and j == 1:
        Nx, Ny = router_list_Nxy[2]

    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_grp5_multicast_rx_core", from_dx,
          from_dy, Nx, Ny, router_list_Nxy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b01
    prim_router.Send_en = 0
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 895
    prim_router.Receive_number = 0
    prim_router.Nx = Nx
    prim_router.Ny = Ny
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 895
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = []
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 896, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
    #                      pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 256
    prim_soma2.length_out = 256
    prim_soma2.length_ciso = 256
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x700 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, prim_soma2


# i！= 0,2,4,6,8,10,12     接收多播数据,并多播到shortcout组
def prim_shortcout_multicast_rx_core(m, n, i, j, phase):

    Nx = Ny = 0
    Cxy = 0

    router_list_recv = [(14, 0), (14, 1), (14, 2),
                        (14, 3), (14, 4), (14, 5), (14, 6)]
    router_list_Nxy = [(1, 0), (0, 0)]

    if i % 2 == 0 and j == 2:
        Cxy = 1
        Nx, Ny = router_list_Nxy[0]
    elif i % 2 == 1 and j == 2:
        Cxy = 0
        Nx, Ny = router_list_Nxy[1]

    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_shortcout_multicast_rx_core",
          from_dx, from_dy, Nx, Ny, router_list_Nxy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = Cxy
    prim_router.Send_en = 0
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 895
    prim_router.Receive_number = 0
    prim_router.Nx = Nx
    prim_router.Ny = Ny
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 895
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 0

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = []
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 896, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
    #                      pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 256
    prim_soma2.length_out = 256
    prim_soma2.length_ciso = 256
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x700 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, prim_soma2


# i==0,2,4,6,8,10,12 发送本core数据，并多播接收到的数据
def prim_grp6_rx_tx_core(m, n, i, j, phase):
    router_list_rhead = [(0, 2), (0, 2), (0, 2), (0, 2),
                         (0, 2), (0, 2), (0, 2)]
    router_list_send = [(0, 5), (2, 5), (4, 5), (6, 5),
                        (8, 5), (10, 5), (12, 5)]
    router_list_recv = [(0, 0), (2, 0), (4, 0), (6, 0),
                        (8, 0), (10, 0), (12, 0)]

    dx, dy = router_list_rhead[i // 2]
    to_dx, to_dy = router_list_send[i // 2]
    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_grp6_rx_tx_core", dx, dy, to_dx, to_dy, from_dx, from_dy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

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
    prim_soma1.Addr_Start_in = 0x4000 + 0x380 * phase  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "prim_grp6_rx_tx_core" + str(phase)
    prim_in = prim_soma1.init_data()

    data = []
    for i in range(len(prim_in[0])):
        # type_in 类型，若int8,3,int32,....
        data.append([phase + 3, phase + 3, phase + 3, phase + 3])

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
    prim_router.Nx = 1
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 447
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [{'core_id': [((m, n), (to_dx, to_dy))], 'data_num': 448,
                                         'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': phase}]
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x380 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, prim_soma1, prim_router, prim_soma2


# i！= 0,2,4,6,8,10,12     接收多播数据,并多播到shortcout组
def prim_grp6_multicast_rx_core(m, n, i, j, phase):

    Nx = Ny = 0
    Cxy = 0

    router_list_recv = [(14, 0), (14, 1), (14, 2),
                        (14, 3), (14, 4), (14, 5), (14, 6)]
    router_list_Nxy = [(0, 1), (-1, 0), (0, 1)]

    if i % 2 == 1 and j == 0:
        Nx, Ny = router_list_Nxy[0]
    elif i % 2 == 1 and j == 1:
        Nx, Ny = router_list_Nxy[1]
    elif i % 2 == 0 and j == 1:
        Nx, Ny = router_list_Nxy[2]

    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_grp5_multicast_rx_core", from_dx,
          from_dy, Nx, Ny, router_list_Nxy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b01
    prim_router.Send_en = 0
    prim_router.Receive_en = 1
    prim_router.Addr_Dout_base = 0x1000
    prim_router.Dout_Mem_sel = 1
    prim_router.Addr_Dout_length = 223
    prim_router.Send_number = 447
    prim_router.Addr_Rhead_base = 0x300
    prim_router.Addr_Rhead_length = 0
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 895
    prim_router.Receive_number = 0
    prim_router.Nx = Nx
    prim_router.Ny = Ny
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 895
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = []
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 896, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
    #                      pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 256
    prim_soma2.length_out = 256
    prim_soma2.length_ciso = 256
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x700 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, prim_soma2


# i==0,2,4,6,8,10,12 发送本core数据，并多播接收到的数据
def prim_grp7_rx_tx_core(m, n, i, j, phase):
    router_list_rhead = [(0, 2), (0, 2), (0, 2), (0, 2),
                         (0, 2), (0, 2), (0, 2)]
    router_list_send = [(0, 7), (2, 7), (4, 7), (6, 7),
                        (8, 7), (10, 7), (12, 7)]
    router_list_recv = [(0, 3), (2, 3), (4, 3), (6, 3),
                        (8, 3), (10, 3), (12, 3)]

    dx, dy = router_list_rhead[i // 2]
    to_dx, to_dy = router_list_send[i // 2]
    from_dx, from_dy = router_list_recv[i // 2]

    print("prim_grp7_rx_tx_core", dx, dy, to_dx, to_dy, from_dx, from_dy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 1

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
    prim_soma1.Addr_Start_in = 0x4000 + 0x380 * phase  # num 不同phase，地址偏移
    prim_soma1.Addr_Start_ciso = 0  #
    prim_soma1.Addr_Start_out = 0x9000  # mem3发送
    prim_soma1.mem_sel = 1
    prim_soma1.in_row_max = 0

    memblock_num = "prim_grp7_rx_tx_core" + str(phase)
    prim_in = prim_soma1.init_data()

    data = []
    for i in range(len(prim_in[0])):
        # type_in 类型，若int8,3,int32,....
        data.append([phase + 3, phase + 3, phase + 3, phase + 3])

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
    prim_router.Nx = 1
    prim_router.Ny = 0
    prim_router.Send_PI_en = 0
    prim_router.Back_sign_en = 0
    prim_router.Send_PI_num = 0
    prim_router.Receive_sign_num = 0
    prim_router.Send_PI_addr_base = 0
    prim_router.Relay_number = 447
    prim_router.Q = 0
    prim_router.Receive_sign_en = 0
    prim_router.T_mode = 1
    prim_router.Soma_in_en = 1

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [{'core_id': [((m, n), (to_dx, to_dy))], 'data_num': 448,
                                         'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': phase}]
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x380 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, prim_soma1, prim_router, None


def prim_grp8_rx_core(m, n, i, j, phase):  # i== 0,2,4,6,8,10,12

    router_list_recv = [(0, 5), (2, 5), (4, 5), (6, 5),
                        (8, 5), (10, 5), (12, 5)]

    from_dx, from_dy = router_list_recv[i // 2]

    print("+++++++++++++++prim_grp8_rx_core", from_dx, from_dy, )

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
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
    prim_router.Soma_in_en = 0

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = []
    prim_router.recv_source_core_grp = [
        {'core_id': [((m, n), (from_dx, from_dy))], 'data_num': 448, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
    #                      pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

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
    prim_soma2.Addr_Start_out = 0x6400 + 0x700 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, prim_soma2


def prim_OB11_tx_core(m, n, i, j, phase):  # i==0,2,4,6,8,10,12 发送本core数据，并多播接收到的数据
    router_list_rhead = [(0, 2), (0, 2), (0, 2), (0, 2)]
    router_list_send = [(8, 1), (9, 1), (10, 1), (11, 1)]

    dx, dy = router_list_rhead[i-8]
    to_dx, to_dy = router_list_send[i-8]

    print("prim_OB11_tx_core", dx, dy, to_dx, to_dy)

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
    prim_axon.A2S2_mode = 0

    prim_router = Prim_09_Router()
    prim_router.Rhead_mode = 1
    prim_router.CXY = 0b00
    prim_router.Send_en = 1
    prim_router.Receive_en = 0
    prim_router.Addr_Dout_base = 0x400
    prim_router.Dout_Mem_sel = 0
    prim_router.Addr_Dout_length = 111
    prim_router.Send_number = 223
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
    prim_router.Soma_in_en = 1

    const = 0xdeadbeef
    prim1_in = []
    for i in range((prim_router.Addr_Dout_length + 1) * 4):
        prim1_in.append([const])

    prim_router.memory_blocks = [
        {'name': 'core1_0memInit',
         'start': prim_router.Addr_Dout_base + 0x8000,
         'data': prim1_in,
         'mode': 0},
    ]

    prim_router.Receive_PI_addr_base = 0x7a0 >> 2

    prim_router.send_destin_core_grp = [{'core_id': [((m, n+1), (to_dx, to_dy))], 'data_num': 224,
                                         'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': phase}]
    prim_router.recv_source_core_grp = []
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
                         pack_per_Rhead=224, A_offset=0, Const=0, EN=1)

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, None


def prim_IB12_rx_core(m, n, i, j, phase):  # i== 0,2,4,6,8,10,12

    router_list_recv = [(8, 9), (9, 9), (10, 9), (11, 9)]

    from_dx, from_dy = router_list_recv[i-8]

    print("prim_IB12_rx_core", from_dx, from_dy, )

    prim_axon = Prim_41_Axon()
    prim_axon.axon_delay = True
    prim_axon.Addr_InA_base = 0x0000
    prim_axon.Addr_InB_base = 0x0000
    prim_axon.Addr_V_base = 0x0000
    prim_axon.L4_num = DELAY_L4_Num
    prim_axon.L5_num = DELAY_L5_Num
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
    prim_router.Addr_Din_base = 0x400
    prim_router.Addr_Din_length = 223
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
        {'core_id': [((m, n-1), (from_dx, from_dy))], 'data_num': 224, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1,
         'sync_phase_num': phase}, ]
    prim_router.instant_prim_request = []
    prim_router.instant_request_back = []

    # prim_router.addRHead(S=0, T=1, P=0, Q=0, X=dx, Y=dy, A=0,
    #                      pack_per_Rhead=447, A_offset=0, Const=0, EN=1)

    # move prim in 1-4 phases of core1
    prim_soma2 = Prim_06_move_merge()
    prim_soma2.length_in = 64
    prim_soma2.length_out = 64
    prim_soma2.length_ciso = 64
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
    prim_soma2.Addr_Start_out = 0x6400 + 0x1C0 * phase  #
    prim_soma2.Addr_Start_ciso = 0x0000  # Null
    prim_soma2.in_row_max = 0

    # # move -> mem3 发送

    # prim_soma2 = None
    return prim_axon, None, prim_router, prim_soma2
# ------------------------------------------------------------------


def prim_IB11__config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_IB11__config", m, n, i, j)
    prims_dict = copy.deepcopy(prims_dict)
    prim_axon, prim_soma1, prim_router, prim_soma2 = prim_IB11_Tx_core(
        m, n, i, j, phase=0)
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = prim_soma1
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = prim_soma2
    prims_list.append(prims_dict)

    return prims_list


def prim_grp5_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_grp5_config", m, n, i, j)

    prims_dict = copy.deepcopy(prims_dict)
    if i % 2 == 0 and j == 0:
        prim_axon, prim_soma1, prim_router, prim_soma2 = prim_grp5_rx_tx_core(
            m, n, i, j, phase=0)
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = prim_soma1
        prims_dict['router'] = prim_router
        prims_dict['soma2'] = prim_soma2
        prims_list.append(prims_dict)
    else:
        prim_axon, prim_soma1, prim_router, prim_soma2 = prim_grp5_multicast_rx_core(
            m, n, i, j, phase=0)
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = prim_soma1
        prims_dict['router'] = prim_router
        prims_dict['soma2'] = prim_soma2
        prims_list.append(prims_dict)

    return prims_list


def prim_grp6_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_grp6_config", m, n, i, j)
    prims_dict = copy.deepcopy(prims_dict)
    if i % 2 == 0 and j == 3:
        prim_axon, prim_soma1, prim_router, prim_soma2 = prim_grp6_rx_tx_core(
            m, n, i, j, phase=0)
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = prim_soma1
        prims_dict['router'] = prim_router
        prims_dict['soma2'] = prim_soma2
        prims_list.append(prims_dict)
    else:
        prim_axon = prim_delay()
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = None
        prims_dict['router'] = None
        prims_dict['soma2'] = None
        prims_list.append(prims_dict)

    return prims_list


def prim_grp7_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_grp7_config", m, n, i, j)
    prims_dict = copy.deepcopy(prims_dict)
    if i % 2 == 0 and j == 5:
        prim_axon, prim_soma1, prim_router, prim_soma2 = prim_grp7_rx_tx_core(
            m, n, i, j, phase=0)
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = prim_soma1
        prims_dict['router'] = prim_router
        prims_dict['soma2'] = prim_soma2
        prims_list.append(prims_dict)
    else:
        prim_axon = prim_delay()
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = None
        prims_dict['router'] = None
        prims_dict['soma2'] = None
        prims_list.append(prims_dict)

    return prims_list


def prim_grp8_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("+++++++++++prim_grp8_config", m, n, i, j)
    prims_dict = copy.deepcopy(prims_dict)
    if i % 2 == 0 and j == 7:
        prim_axon, prim_soma1, prim_router, prim_soma2 = prim_grp8_rx_core(
            m, n, i, j, phase=0)
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = prim_soma1
        prims_dict['router'] = prim_router
        prims_dict['soma2'] = prim_soma2
        prims_list.append(prims_dict)
    else:
        prim_axon = prim_delay()
        prims_dict['axon'] = prim_axon
        prims_dict['soma1'] = None
        prims_dict['router'] = None
        prims_dict['soma2'] = None
        prims_list.append(prims_dict)

    return prims_list


def prim_OB11_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_OB11_config", m, n, i, j)
    prim_axon, prim_soma1, prim_router, prim_soma2 = prim_OB11_tx_core(
        m, n, i, j, phase=0)
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = prim_soma1
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = prim_soma2
    prims_list.append(prims_dict)

    return prims_list


def prim_IB12_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_IB12_config", m, n, i, j)
    prim_axon, prim_soma1, prim_router, prim_soma2 = prim_IB12_rx_core(
        m, n, i, j, phase=0)
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = prim_soma1
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = prim_soma2
    prims_list.append(prims_dict)

    return prims_list


def prim_shortcut_config(m, n, i, j):
    prims_list = []
    prims_dict = {}

    print("prim_shortcut_config", m, n, i, j)
    prims_dict = copy.deepcopy(prims_dict)
    prim_axon, prim_soma1, prim_router, prim_soma2 = prim_shortcout_multicast_rx_core(
        m, n, i, j, phase=0)
    prims_dict['axon'] = prim_axon
    prims_dict['soma1'] = prim_soma1
    prims_dict['router'] = prim_router
    prims_dict['soma2'] = prim_soma2
    prims_list.append(prims_dict)

    return prims_list


# ---------------------------------------------------------------------------


def registers_dict(m, n, i, j):
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

    return registers_dict


def instant_prims_list(m, n, i, j):
    instant_prims_list = []
    return instant_prims_list


def prims_list(m, n, i, j):
    prims_list = []

    if m == 1 and n == 1:
        if i == 14:
            prims_list = prim_IB11__config(1, 1, 14, j)
        elif j == 0 or j == 1:
            prims_list = prim_grp5_config(1, 1, i, j)
        elif j == 2:
            prims_list = prim_shortcut_config(1, 1, i, j)
        elif j == 3 or j == 4:
            prims_list = prim_grp6_config(1, 1, i, j)
        elif j == 5 or j == 6:
            prims_list = prim_grp7_config(1, 1, i, j)
        elif j == 7 or j == 8:
            prims_list = prim_grp8_config(1, 1, i, j)
        elif j == 9:
            prims_list = prim_OB11_config(1, 1, i, 9)
    elif m == 1 and n == 2:
        prims_list = prim_IB12_config(1, 2, i, 1)

    return prims_list


def core_config(m, n, i, j):
    print("core_config", m, n, i, j)
    core_config = {}
    core_config['prims'] = prims_list(m, n, i, j)
    core_config['instant_prims'] = instant_prims_list(m, n, i, j)
    core_config['registers'] = registers_dict(m, n, i, j)
    return core_config


def group0_config():  # IB11
    group0_config = {}
    group0_config['clock'] = GRP0_CLOCK
    group0_config['mode'] = 1
    # IB11
    for j in range(0, 7):
        group0_config[((1, 1), (14, j))] = core_config(1, 1, 14, j)
        print("group0_config", j)

    return group0_config


def group1_config():  # Grp5
    group1_config = {}
    group1_config['clock'] = GRP1_CLOCK
    group1_config['mode'] = 1
    # Group5
    for i in range(0, 14):
        for j in range(0, 2):
            group1_config[((1, 1), (i, j))] = core_config(1, 1, i, j)
            print("group1_config", i, j)

    return group1_config


def group2_config():  # Grp8
    group2_config = {}
    group2_config['clock'] = GRP2_CLOCK
    group2_config['mode'] = 1
    # Group8
    for i in range(0, 14, 2):
        group2_config[((1, 1), (i, 7))] = core_config(1, 1, i, 7)
        print("+++++++++++++group2_config", i, 7)

    return group2_config


def group6_config():  # Grp8
    group6_config = {}
    group6_config['clock'] = GRP2_CLOCK
    group6_config['mode'] = 1
    # Group8
    for i in range(0, 14, 2):
        group6_config[((1, 1), (i, 3))] = core_config(1, 1, i, 3)
        print("group6_config", i, 3)

    return group6_config


def group7_config():  # Grp8
    group7_config = {}
    group7_config['clock'] = GRP2_CLOCK
    group7_config['mode'] = 1
    # Group8
    for i in range(0, 14, 2):
        group7_config[((1, 1), (i, 5))] = core_config(1, 1, i, 5)
        print("group7_config", i, 5)

    return group7_config


def group3_config():  # OB11
    group3_config = {}
    group3_config['clock'] = GRP3_CLOCK
    group3_config['mode'] = 1
    # OB11

    for i in range(8, 12):
        group3_config[((1, 1), (i, 9))] = core_config(1, 1, i, 9)
        print("group3_config", i)

    return group3_config


def group4_config():  # IB12

    group4_config = {}
    group4_config['clock'] = GRP4_CLOCK
    group4_config['mode'] = 1
    # IB12
    for i in range(8, 12):
        group4_config[((1, 2), (i, 1))] = core_config(1, 2, i, 1)
        print("group4_config", i, 1)

    return group4_config


def group5_config():  # ShortCut
    group5_config = {}
    group5_config['clock'] = GRP4_CLOCK
    group5_config['mode'] = 1
    # ShortCut
    for i in range(0, 14):  # 0~13
        group5_config[((1, 1), (i, 2))] = core_config(1, 1, i, 2)
        print("group5_config", i, 2)

    return group5_config


# 映射策略\
map_config = {
    'sim_clock': SIM_CLOCK,
    # "step_clock": {
    #     ((1, 1), 0): (SIM_CLOCK, SIM_CLOCK),
    #     ((1, 2), 0): (SIM_CLOCK, SIM_CLOCK),
    # },
    ((1, 1), 0): {
        "step_exe_number": 1,

        0: group0_config(),  # IB11
        1: group1_config(),  # Grp5
        2: group2_config(),  # Grp8
        3: group3_config(),  # OB11
        4: group5_config(),  # ShortCut
        5: group6_config(),  # Grp6
        6: group7_config()  # Grp7
    },
    ((1, 2), 0): {
        "step_exe_number": 1,
        0: group4_config(),  # IB12 in chip(1,2)
    }

}

map_config[((1, 1), 0)].pop(0, None)
map_config[((1, 1), 0)].pop(1, None)
map_config[((1, 1), 0)].pop(2, None)
map_config[((1, 1), 0)].pop(4, None)
map_config[((1, 1), 0)].pop(5, None)
map_config[((1, 1), 0)].pop(6, None)

# for step_id, group_id, config in MapConfig(map_config):
#     phase_num = None
#     if isinstance(step_id, str):
#         continue
#     for id in config.core_list:
#         config._core_cofig[id]["prims"] = config._core_cofig[id]["prims"][0:1]
map_config['sim_clock'] = 20000
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

# # 测试配置

test_config = {
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    'debug_file_switch': HardwareDebugFileSwitch().close_all.singla_chip.dict,
    'test_group_phase': test_phase  # (phase_group, phase_num)
}


# 测试配置
# from generator.test_engine.test_config import HardwareDebugFileSwitch
# test_config = {
#     'tb_name': tb_name,
#     'test_mode': TestMode.MEMORY_STATE,
#     'test_group_phase': test_phase  # (phase_group, phase_num)
# }


# 开始测试

def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
