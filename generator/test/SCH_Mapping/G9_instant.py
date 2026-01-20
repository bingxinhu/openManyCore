
import numpy as np


from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router, Prim_06_move_merge
from generator.test_engine import TestMode, TestEngine
SIMPATH = 'temp\\out_files\\'
tb_name = 'M99999'
np.random.seed(101)

def Gen_G9_instant(start_instant_PI_num = 0, init = True):

    map_config = {
        'sim_clock': 80000,
        0: {
            # "cycles_number": 1,
            0: {'clock': 80000,     # 有静态phase
                'mode': 1,
                },
            1:  {'clock': 80000,
                 'mode': 1,
                 }
        }
    }

    M = 8
    chip = (0, 0)

    # 低两行
    for core_x in range(M):
        for core_y in range(2):
            if core_x == 0 and core_y == 0:
                Router_Prim_1 = Prim_09_Router()
                Router_Prim_1.Rhead_mode = 1
                Router_Prim_1.CXY = 0b00
                Router_Prim_1.Send_en = 1
                Router_Prim_1.Receive_en = 0
                Router_Prim_1.Addr_Dout_base = 0x380
                Router_Prim_1.Dout_Mem_sel = 0
                Router_Prim_1.Addr_Dout_length = 783
                Router_Prim_1.Send_number = 1567
                Router_Prim_1.Addr_Rhead_base = 0x328
                Router_Prim_1.Addr_Rhead_length = 1
                Router_Prim_1.Addr_Din_base = 0x800
                Router_Prim_1.Addr_Din_length = 0
                Router_Prim_1.Receive_number = 0
                Router_Prim_1.Nx = 0
                Router_Prim_1.Ny = 0
                Router_Prim_1.Send_PI_en = 0
                Router_Prim_1.Back_sign_en = 1
                Router_Prim_1.Send_PI_num = 0
                Router_Prim_1.Receive_sign_num = 0
                Router_Prim_1.Receive_PI_addr_base = 0x350 >> 2
                Router_Prim_1.Relay_number = 0
                Router_Prim_1.Q = 1
                Router_Prim_1.Receive_sign_en = 0
                Router_Prim_1.T_mode = 1
                Router_Prim_1.Soma_in_en = 0

                prim0_in = []
                # const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                for p in range((Router_Prim_1.Addr_Dout_length+1)*4):
                    const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                    prim0_in.append([const])
                if init:
                    Router_Prim_1.memory_blocks = [
                        {'name': 'core0_0memInit',
                            'start': Router_Prim_1.Addr_Dout_base+0x8000,
                            'data': prim0_in,
                            'mode': 0},
                    ]

                # Router_Prim_1.Receive_PI_addr_base = 0x7a0 >> 2
                dst_y = core_x // 2 + 2
                if core_y == 0:
                    A = 0
                else:
                    A = 7 * 14 * 32 // 8

                if core_x % 2 == 0:
                    dst_x = 0
                else:
                    dst_x = 4

                Router_Prim_1.send_destin_core_grp = [
                    {"core_id": (chip, (dst_x + 0, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 1, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 2, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 3, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 0, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 1, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 2, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 3, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)

                Router_Prim_1.recv_source_core_grp = []
                Router_Prim_1.instant_prim_request = []
                Router_Prim_1.instant_request_back = []
                for j in range(8):
                    for q in range(2, 6):
                        Router_Prim_1.instant_request_back.append((chip, (j, q)))

                map_config[0][1][(chip, (core_x, core_y))] = {}
                map_config[0][1][(chip, (core_x, core_y))]['prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None, 'soma2': None})
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_1, 'soma2': None})
                map_config[0][1][(chip, (core_x, core_y))]['registers'] = {
                    "Receive_PI_addr_base": 0x350 >> 2,
                    "PI_CXY": 1,
                    "PI_Nx": 1,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": start_instant_PI_num
                }
            elif core_x == 0 and core_y == 1:
                Router_Prim_2 = Prim_09_Router()
                Router_Prim_2.Rhead_mode = 1
                Router_Prim_2.CXY = 0b00
                Router_Prim_2.Send_en = 1
                Router_Prim_2.Receive_en = 0
                Router_Prim_2.Addr_Dout_base = 0x380
                Router_Prim_2.Dout_Mem_sel = 0
                Router_Prim_2.Addr_Dout_length = 783
                Router_Prim_2.Send_number = 1567
                Router_Prim_2.Addr_Rhead_base = 0x328
                Router_Prim_2.Addr_Rhead_length = 1
                Router_Prim_2.Addr_Din_base = 0x800
                Router_Prim_2.Addr_Din_length = 0
                Router_Prim_2.Receive_number = 0
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
                Router_Prim_2.Receive_PI_addr_base = 0x350 >> 2

                # const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                prim1_in = []
                for p in range((Router_Prim_2.Addr_Dout_length+1)*4):
                    const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                    prim1_in.append([const])
                if init:
                    Router_Prim_2.memory_blocks = [
                        {'name': 'core1_0memInit',
                            'start': Router_Prim_2.Addr_Dout_base+0x8000,
                            'data': prim1_in,
                            'mode': 0},
                    ]

                # Router_Prim_2.Receive_PI_addr_base = 0x7a0 >> 2
                dst_y = core_x // 2 + 2
                if core_y == 0:
                    A = 0
                else:
                    A = 7 * 14 * 32 // 8

                if core_x % 2 == 0:
                    dst_x = 0
                else:
                    dst_x = 4

                Router_Prim_2.send_destin_core_grp = [
                    {"core_id": (chip, (dst_x + 0, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 1, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 2, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 3, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 0, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 1, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 2, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 3, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)

                Router_Prim_2.recv_source_core_grp = []
                Router_Prim_2.instant_prim_request = []
                Router_Prim_2.instant_request_back = []

                map_config[0][1][(chip, (core_x, core_y))] = {}
                map_config[0][1][(chip, (core_x, core_y))]['prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None, 'soma2': None})
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_2, 'soma2': None})
                map_config[0][1][(chip, (core_x, core_y))]['registers'] = {
                    "Receive_PI_addr_base": 0x350 >> 2,
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
                    "start_instant_PI_num": start_instant_PI_num
                }
            else:
                Router_Prim_2 = Prim_09_Router()
                Router_Prim_2.Rhead_mode = 1
                Router_Prim_2.CXY = 0b00
                Router_Prim_2.Send_en = 1
                Router_Prim_2.Receive_en = 0
                Router_Prim_2.Addr_Dout_base = 0x380
                Router_Prim_2.Dout_Mem_sel = 0
                Router_Prim_2.Addr_Dout_length = 783
                Router_Prim_2.Send_number = 1567
                Router_Prim_2.Addr_Rhead_base = 0x328
                Router_Prim_2.Addr_Rhead_length = 1
                Router_Prim_2.Addr_Din_base = 0x800
                Router_Prim_2.Addr_Din_length = 0
                Router_Prim_2.Receive_number = 0
                Router_Prim_2.Nx = 0
                Router_Prim_2.Ny = 0
                Router_Prim_2.Send_PI_en = 0
                Router_Prim_2.Back_sign_en = 0
                Router_Prim_2.Send_PI_num = 0
                Router_Prim_2.Receive_sign_num = 0
                Router_Prim_2.Send_PI_addr_base = 0
                Router_Prim_2.Relay_number = 0
                Router_Prim_2.Q = 1
                Router_Prim_2.Receive_sign_en = 0
                Router_Prim_2.T_mode = 1
                Router_Prim_2.Soma_in_en = 0
                Router_Prim_2.Receive_PI_addr_base = 0x350 >> 2

                # const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                prim1_in = []
                for p in range((Router_Prim_2.Addr_Dout_length + 1) * 4):
                    const = np.random.randint(-2 ** 31, 2 ** 31 - 1)
                    prim1_in.append([const])
                if init:
                    Router_Prim_2.memory_blocks = [
                        {'name': 'core1_0memInit',
                         'start': Router_Prim_2.Addr_Dout_base + 0x8000,
                         'data': prim1_in,
                         'mode': 0},
                    ]

                # Router_Prim_2.Receive_PI_addr_base = 0x7a0 >> 2
                dst_y = core_x // 2 + 2
                if core_y == 0:
                    A = 0
                else:
                    A = 7 * 14 * 32 // 8

                if core_x % 2 == 0:
                    dst_x = 0
                else:
                    dst_x = 4

                Router_Prim_2.send_destin_core_grp = [
                    {"core_id": (chip, (dst_x + 0, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 1, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 2, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (dst_x + 3, dst_y)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 0, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 1, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 2, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
                Router_Prim_2.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 3, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)

                Router_Prim_2.recv_source_core_grp = []
                Router_Prim_2.instant_prim_request = []
                Router_Prim_2.instant_request_back = []

                map_config[0][1][(chip, (core_x, core_y))] = {}
                map_config[0][1][(chip, (core_x, core_y))]['prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None, 'soma2': None})
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'] = []
                map_config[0][1][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_2, 'soma2': None})
                if core_y == 0:
                    if core_x == M - 1:
                        Nx = 0
                        Ny = 1
                    else:
                        Nx = 1
                        Ny = 0
                else:
                    Nx = -1
                    Ny = 0
                map_config[0][1][(chip, (core_x, core_y))]['registers'] = {
                    "Receive_PI_addr_base": 0x350 >> 2,
                    "PI_CXY": 1,
                    "PI_Nx": Nx,
                    "PI_Ny": Ny,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 1,
                    "fixed_instant_PI": 1,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": start_instant_PI_num
                }


    N = 8
    P = 4
    for core_x in range(N):
        for core_y in range(2, 2 + P):
            soma2 = Prim_06_move_merge()
            soma2.length_in = 14*32
            soma2.length_ciso = 16
            soma2.num_in = 14
            soma2.num_ciso = 14
            soma2.length_out = 14*32
            soma2.num_out = 14
            soma2.type_in = 1
            soma2.type_out = 1
            soma2.in_cut_start = 0
            soma2.Reset_Addr_in = 1
            soma2.Reset_Addr_out = 1
            soma2.Reset_Addr_ciso = 1
            soma2.Row_ck_on = 0
            soma2.Addr_Start_in = 0x8380
            soma2.Addr_Start_ciso = 0x10000 >> 2
            soma2.Addr_Start_out = 0xe780 >> 2
            soma2.in_row_max = 0
            soma2.mem_sel = 0

            if core_x == 0 and core_y == 2:
                Router_Prim_3 = Prim_09_Router()
                Router_Prim_3.Rhead_mode = 1
                Router_Prim_3.CXY = 0b00
                Router_Prim_3.Send_en = 0
                Router_Prim_3.Receive_en = 1
                Router_Prim_3.Addr_Dout_base = 0x400
                Router_Prim_3.Dout_Mem_sel = 0
                Router_Prim_3.Addr_Dout_length = 24
                Router_Prim_3.Send_number = 0
                Router_Prim_3.Addr_Rhead_base = 0x300
                Router_Prim_3.Addr_Rhead_length = 0
                Router_Prim_3.Addr_Din_base = 0x380
                Router_Prim_3.Addr_Din_length = 783
                Router_Prim_3.Receive_number = 1
                Router_Prim_3.Nx = 0
                Router_Prim_3.Ny = 0
                Router_Prim_3.Send_PI_en = 1
                Router_Prim_3.Back_sign_en = 0
                Router_Prim_3.Send_PI_num = 0
                Router_Prim_3.Receive_sign_num = 0
                Router_Prim_3.Send_PI_addr_base = 0x350 >> 2
                Router_Prim_3.Relay_number = 0
                Router_Prim_3.Q = 0
                Router_Prim_3.Receive_sign_en = 1
                Router_Prim_3.T_mode = 1
                Router_Prim_3.Soma_in_en = 0
                #Router_Prim_3.Receive_PI_addr_base = 0x350 >> 2

                src_x = core_x // 4 + (core_y - 2) * 2
                src_y = 0
                Router_Prim_3.send_destin_core_grp = []
                Router_Prim_3.recv_source_core_grp = [
                    {"core_id": (chip, (src_x, src_y + 0)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (src_x, src_y + 1)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_3.instant_prim_request = [((chip, (0, 0)), 0)]
                Router_Prim_3.instant_request_back = []
                Router_Prim_3.add_instant_pi(
                    PI_addr_offset=0, A_valid=0, S1_valid=0, R_valid=1, S2_valid=0, X=0, Y=-2, Q=1)

                map_config[0][0][(chip, (core_x, core_y))] = {}
                map_config[0][0][(chip, (core_x, core_y))]['prims'] = [{'axon': None, 'soma1': None, 'router': Router_Prim_3, 'soma2': soma2}]
                #map_config[0][0][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_3, 'soma2': soma2})
                map_config[0][0][(chip, (core_x, core_y))]['registers'] = {
                    "Receive_PI_addr_base": 0,
                     "PI_CXY": 0,
                     "PI_Nx": 0,
                     "PI_Ny": 0,
                     "PI_sign_CXY": 1,
                     "PI_sign_Nx": 1,
                     "PI_sign_Ny": 0,
                     "instant_PI_en": 0,
                     "fixed_instant_PI": 0,
                     "instant_PI_number": 0,
                     "PI_loop_en": 0,
                     "start_instant_PI_num": 0,
                     "Addr_instant_PI_base":0
                }

            elif (P % 2 == 1 and core_x == N - 1 and core_y == P + 1) or (P % 2 == 0 and core_x == 0 and core_y == P + 1):

                Router_Prim_5 = Prim_09_Router()
                Router_Prim_5.Rhead_mode = 1
                Router_Prim_5.CXY = 0b00
                Router_Prim_5.Send_en = 0
                Router_Prim_5.Receive_en = 1
                Router_Prim_5.Addr_Dout_base = 0x400
                Router_Prim_5.Dout_Mem_sel = 0
                Router_Prim_5.Addr_Dout_length = 63
                Router_Prim_5.Send_number = 127
                Router_Prim_5.Addr_Rhead_base = 0x700
                Router_Prim_5.Addr_Rhead_length = 0
                Router_Prim_5.Addr_Din_base = 0x380
                Router_Prim_5.Addr_Din_length = 783
                Router_Prim_5.Receive_number = 1
                Router_Prim_5.Nx = 0
                Router_Prim_5.Ny = 0
                Router_Prim_5.Send_PI_en = 0
                Router_Prim_5.Back_sign_en = 0
                Router_Prim_5.Send_PI_num = 0
                Router_Prim_5.Receive_sign_num = 0
                Router_Prim_5.Send_PI_addr_base = 0
                Router_Prim_5.Relay_number = 0
                Router_Prim_5.Q = 0
                Router_Prim_5.Receive_sign_en = 1
                Router_Prim_5.T_mode = 1
                Router_Prim_5.Soma_in_en = 0
                # Router_Prim_5.Receive_PI_addr_base = 0x350 >> 2

                src_x = core_x // 4 + (core_y - 2) * 2
                src_y = 0
                Router_Prim_5.send_destin_core_grp = []
                Router_Prim_5.recv_source_core_grp = [
                    {"core_id": (chip, (src_x, src_y + 0)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (src_x, src_y + 1)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_5.instant_prim_request = []
                Router_Prim_5.instant_request_back = []

                map_config[0][0][(chip, (core_x, core_y))] = {}
                map_config[0][0][(chip, (core_x, core_y))]['prims'] = [{'axon': None, 'soma1': None, 'router': Router_Prim_5, 'soma2': soma2}]
                # map_config[0][0][(chip, (core_x, core_y))]['instant_prims'] = []
                # map_config[0][0][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_5, 'soma2': soma2})
                map_config[0][0][(chip, (core_x, core_y))]['registers'] = {
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
                     "start_instant_PI_num": 0,
                     "Addr_instant_PI_base":0
                }

            else:
                Router_Prim_5 = Prim_09_Router()
                Router_Prim_5.Rhead_mode = 1
                Router_Prim_5.CXY = 0b00
                Router_Prim_5.Send_en = 0
                Router_Prim_5.Receive_en = 1
                Router_Prim_5.Addr_Dout_base = 0x400
                Router_Prim_5.Dout_Mem_sel = 0
                Router_Prim_5.Addr_Dout_length = 63
                Router_Prim_5.Send_number = 127
                Router_Prim_5.Addr_Rhead_base = 0x700
                Router_Prim_5.Addr_Rhead_length = 0
                Router_Prim_5.Addr_Din_base = 0x380
                Router_Prim_5.Addr_Din_length = 783
                Router_Prim_5.Receive_number = 1
                Router_Prim_5.Nx = 0
                Router_Prim_5.Ny = 0
                Router_Prim_5.Send_PI_en = 0
                Router_Prim_5.Back_sign_en = 0
                Router_Prim_5.Send_PI_num = 0
                Router_Prim_5.Receive_sign_num = 0
                Router_Prim_5.Send_PI_addr_base = 0
                Router_Prim_5.Relay_number = 0
                Router_Prim_5.Q = 0
                Router_Prim_5.Receive_sign_en = 1
                Router_Prim_5.T_mode = 1
                Router_Prim_5.Soma_in_en = 0
                # Router_Prim_5.Receive_PI_addr_base = 0x350 >> 2

                src_x = core_x // 4 + (core_y - 2) * 2
                src_y = 0
                Router_Prim_5.send_destin_core_grp = []
                Router_Prim_5.recv_source_core_grp = [
                    {"core_id": (chip, (src_x, src_y + 0)), "data_num": 392, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": (chip, (src_x, src_y + 1)), "data_num": 392, "T_mode": 1, "Rhead_num": 1}
                ]
                Router_Prim_5.instant_prim_request = []
                Router_Prim_5.instant_request_back = []

                map_config[0][0][(chip, (core_x, core_y))] = {}
                map_config[0][0][(chip, (core_x, core_y))]['prims'] = [{'axon': None, 'soma1': None, 'router': Router_Prim_5, 'soma2': soma2}]
                # map_config[0][0][(chip, (core_x, core_y))]['instant_prims'] = []
                # map_config[0][0][(chip, (core_x, core_y))]['instant_prims'].append({'axon': None, 'soma1': None, 'router': Router_Prim_5, 'soma2': soma2})
                if core_y % 2 == 0:
                    if core_x == N - 1:
                        Nx = 0
                        Ny = 1
                    else:
                        Nx = 1
                        Ny = 0
                else:
                    if core_x == 0:
                        Nx = 0
                        Ny = 1
                    else:
                        Nx = -1
                        Ny = 0
                map_config[0][0][(chip, (core_x, core_y))]['registers'] = {
                    "Receive_PI_addr_base": 0,
                    "PI_CXY": 0,
                    "PI_Nx": 0,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 1,
                    "PI_sign_Nx": Nx,
                    "PI_sign_Ny": Ny,
                    "instant_PI_en": 0,
                    "fixed_instant_PI": 0,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "start_instant_PI_num": 0,
                    "Addr_instant_PI_base": 0
                }

    return map_config

run = False
if run:
    map_config = Gen_G9_instant(init=True, start_instant_PI_num=0)

    from generator.test_engine.test_config import HardwareDebugFileSwitch
    # 测试配置
    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1), (1, 1)]
    }

    # 开始测试
    #
    # import pickle
    #
    # with open('sch_test_instant_0726.map_config', 'wb') as f:
    #     pickle.dump(map_config, f)

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()

