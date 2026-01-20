from generator.test_engine.test_config import HardwareDebugFileSwitch
import pytest
import numpy as np
import os

from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig
SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
tb_name = "I11001"
np.random.seed(sum(ord(c) for c in tb_name))


def Group0_C00_phas1_temp():
    prim_axon = Prim_41_Axon()
    prim_axon.InA_type = 1  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
    prim_axon.InB_type = 1
    prim_axon.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
    prim_axon.pad_on = False
    prim_axon.Input_fm_Px = 1
    prim_axon.Input_fm_Py = 28
    prim_axon.conv_Kx = 1
    prim_axon.conv_Ky = 1
    prim_axon.conv_Sx = 1
    prim_axon.conv_Sy = 1
    prim_axon.conv_Ex = 1
    prim_axon.conv_Ey = 1
    prim_axon.pad_top = 0
    prim_axon.pad_down = 0
    prim_axon.pad_left = 0
    prim_axon.pad_right = 0
    prim_axon.cin = 256
    prim_axon.cout = 32
    prim_axon.Bias_length = 0
    prim_axon.Reset_Addr_A = 1
    prim_axon.Reset_Addr_V = 1
    prim_axon.Addr_InA_base = 0x6400  # Xp
    prim_axon.Addr_InB_base = 0x0000  # W1
    prim_axon.Addr_Bias_base = 0x0000
    prim_axon.Addr_V_base = 0x2000
    prim_axon.A2S2_mode = 0

    prim_in = prim_axon.init_data()
    prim_axon.memory_blocks = [
        {'name': 'Group0_C00_phas1_Xp',
         'start': prim_axon.Addr_InA_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
         'mode': 0,
         'initialize': True},
        {'name': 'Group0_C00_phas1_W',
         'start': prim_axon.Addr_InB_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[1],
         'mode': 0,
         'initialize': True}
    ]

    return prim_axon


def Group0_C00_phas1():
    prim_axon = Prim_41_Axon()
    prim_axon.InA_type = 1  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
    prim_axon.InB_type = 1
    prim_axon.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
    prim_axon.pad_on = False
    prim_axon.Input_fm_Px = 2
    prim_axon.Input_fm_Py = 28
    prim_axon.conv_Kx = 1
    prim_axon.conv_Ky = 1
    prim_axon.conv_Sx = 1
    prim_axon.conv_Sy = 1
    prim_axon.conv_Ex = 1
    prim_axon.conv_Ey = 1
    prim_axon.pad_top = 0
    prim_axon.pad_down = 0
    prim_axon.pad_left = 0
    prim_axon.pad_right = 0
    prim_axon.cin = 256
    prim_axon.cout = 32
    prim_axon.Bias_length = 0
    prim_axon.Reset_Addr_A = 1
    prim_axon.Reset_Addr_V = 1
    prim_axon.Addr_InA_base = 0x4000  # Xp
    prim_axon.Addr_InB_base = 0x0000  # W1
    prim_axon.Addr_Bias_base = 0x3000
    prim_axon.Addr_V_base = 0x2000
    prim_axon.A2S2_mode = 1

    # const = 5
    # prim0_in = []
    # for i in range(28*2*32):
    #     tmp = []
    #     for j in range(4):
    #         if i == 0 or i == 1:
    #             tmp.append(const)
    #         else:
    #             tmp.append(np.random.randint(255))
    #     prim0_in.append(tmp)
    #
    #
    # const = 8
    # prim1_in = []
    # for i in range(32 * 32):
    #     tmp = []
    #     for j in range(4):
    #         if i == 0 or i == 1:
    #             tmp.append(const)
    #         else:
    #             tmp.append(np.random.randint(255))
    #     prim1_in.append(tmp)
    #
    # prim_axon.memory_blocks = [
    #     {'name': 'core0_0memInit_xp',
    #      'start': prim_axon.Addr_InA_base,
    #      'data': prim0_in,
    #      'mode': 0},
    #     {'name': 'core0_0memInit_w0',
    #      'start': prim_axon.Addr_InB_base,
    #      'data': prim1_in,
    #      'mode': 0},
    # ]

    prim_in = prim_axon.init_data()
    prim_axon.memory_blocks = [
        {'name': 'Sector1_Core_phase1_Xp',
         'start': prim_axon.Addr_InA_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
         'mode': 0,
         'initialize': True},
        {'name': 'Sector1_Core_phase1_W1',
         'start': prim_axon.Addr_InB_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[1],
         'mode': 0,
         'initialize': True}
    ]
    # Relu:
    prim_soma1 = Prim_X5_Soma()
    prim_soma1.type_in = 0  # 可配置更改：[00]int32[01]int8 [11]Tenary
    prim_soma1.type_out = 1
    prim_soma1.cin = 32
    prim_soma1.cout = 32
    prim_soma1.pad_on = False
    prim_soma1.CMP_C_en = 1
    prim_soma1.Input_fm_Px = 2
    prim_soma1.Input_fm_Py = 28
    prim_soma1.pooling_Kx = 1
    prim_soma1.pooling_Ky = 1
    prim_soma1.pooling_Sx = 1
    prim_soma1.pooling_Sy = 1
    prim_soma1.pad_top = 0
    prim_soma1.pad_down = 0
    prim_soma1.pad_left = 0
    prim_soma1.pad_right = 0
    prim_soma1.CMP_C = 0
    prim_soma1.in_cut_start = 0
    prim_soma1.reset_Addr_in = 1
    prim_soma1.reset_Addr_out = 1
    prim_soma1.Row_ck_on = 1
    prim_soma1.Addr_Start_in = 0x2000
    prim_soma1.Addr_Start_out = 0x8400
    prim_soma1.in_row_max = 1  # 2 lines
    prim_soma1.mem_sel = 0
    # phase_grp0;core(0,0) 静态原语phase1

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

    # const = 5
    # prim0_in = []
    # for i in range((Router_Prim_1.Addr_Dout_length + 1) * 4):
    #     tmp = []
    #     for j in range(4):
    #         if i == 0 or i == 1:
    #             tmp.append(const)
    #         else:
    #             tmp.append(np.random.randint(255))
    #     prim0_in.append(tmp)
    #
    # Router_Prim_1.memory_blocks = [
    #     {'name': 'core0_0memInit',
    #      'start': Router_Prim_1.Addr_Dout_base + 0x8000,
    #      'data': prim0_in,
    #      'mode': 0},
    # ]

    Router_Prim_1.Receive_PI_addr_base = 0x7a0 >> 2

    Router_Prim_1.send_destin_core_grp = [
        {"core_id": ((0, 0), (1, 0)), "data_num": 224, "T_mode": 1, "Rhead_num": 1}]
    Router_Prim_1.recv_source_core_grp = []
    Router_Prim_1.instant_prim_request = [(((0, 0), (1, 0)), 0)]
    Router_Prim_1.instant_request_back = []

    Router_Prim_1.add_instant_pi(
        PI_addr_offset=0, A_valid=0, S1_valid=0, R_valid=1, S2_valid=0, X=1, Y=0, Q=0)
    Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                           pack_per_Rhead=223, A_offset=0, Const=0, EN=1)

    # prim_axon = None
    # Router_Prim_1=None

    return prim_axon, prim_soma1, Router_Prim_1


def Group1_C10_phas1():
    prim_axon = Prim_41_Axon()
    prim_axon.InA_type = 1  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
    prim_axon.InB_type = 1
    prim_axon.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
    prim_axon.pad_on = False
    prim_axon.Input_fm_Px = 4
    prim_axon.Input_fm_Py = 28
    prim_axon.conv_Kx = 1
    prim_axon.conv_Ky = 1
    prim_axon.conv_Sx = 1
    prim_axon.conv_Sy = 1
    prim_axon.conv_Ex = 1
    prim_axon.conv_Ey = 1
    prim_axon.pad_top = 0
    prim_axon.pad_down = 0
    prim_axon.pad_left = 0
    prim_axon.pad_right = 0
    prim_axon.cin = 256
    prim_axon.cout = 32
    prim_axon.Bias_length = 0
    prim_axon.Reset_Addr_A = 1
    prim_axon.Reset_Addr_V = 1
    prim_axon.Addr_InA_base = 0x6400  # Xp
    prim_axon.Addr_InB_base = 0x0000  # W1
    prim_axon.Addr_Bias_base = 0x0000
    prim_axon.Addr_V_base = 0x2000
    prim_axon.A2S2_mode = 0

    prim_in = prim_axon.init_data()
    prim_axon.memory_blocks = [
        {'name': 'Sector1_Core_phase1_Xp',
         'start': prim_axon.Addr_InA_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[0],
         'mode': 0,
         'initialize': True},
        {'name': 'Sector1_Core_phase1_W1',
         'start': prim_axon.Addr_InB_base,
         # 'length': 672,   #   流水（不一致的时候）的时候需要声明
         'data': prim_in[1],
         'mode': 0,
         'initialize': True}
    ]

    # Relu:
    prim_soma1 = Prim_X5_Soma()
    prim_soma1.type_in = 0  # 可配置更改：[00]int32[01]int8 [11]Tenary
    prim_soma1.type_out = 1
    prim_soma1.cin = 32
    prim_soma1.cout = 32
    prim_soma1.pad_on = False
    prim_soma1.CMP_C_en = 1
    prim_soma1.Input_fm_Px = 4
    prim_soma1.Input_fm_Py = 28
    prim_soma1.pooling_Kx = 1
    prim_soma1.pooling_Ky = 1
    prim_soma1.pooling_Sx = 1
    prim_soma1.pooling_Sy = 1
    prim_soma1.pad_top = 0
    prim_soma1.pad_down = 0
    prim_soma1.pad_left = 0
    prim_soma1.pad_right = 0
    prim_soma1.CMP_C = 0
    prim_soma1.in_cut_start = 0
    prim_soma1.reset_Addr_in = 1
    prim_soma1.reset_Addr_out = 1
    prim_soma1.Row_ck_on = 1
    prim_soma1.Addr_Start_in = 0x2000
    prim_soma1.Addr_Start_out = 0x4000
    prim_soma1.in_row_max = 2  # 2 lines

    # prim_soma1 = None
    # prim_axon = None
    return prim_axon, prim_soma1


def Group1_C10_instantPI_1():
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
    Router_Prim_3.Addr_Rhead_base = 0x300
    Router_Prim_3.Addr_Rhead_length = 0
    Router_Prim_3.Addr_Din_base = 0x500
    Router_Prim_3.Addr_Din_length = 223
    Router_Prim_3.Receive_number = 0
    Router_Prim_3.Nx = 0
    Router_Prim_3.Ny = 0
    Router_Prim_3.Send_PI_en = 0
    Router_Prim_3.Back_sign_en = 1
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
        {"core_id": ((0, 0), (0, 0)), "data_num": 224, "T_mode": 1, "Rhead_num": 1}]
    Router_Prim_3.instant_prim_request = []
    Router_Prim_3.instant_request_back = [((0, 0), (0, 0))]

    return Router_Prim_3


# --------------------------------------------
prim_axon_0, prim_soma1, Router_Prim_1 = Group0_C00_phas1()
Router_Prim_3 = Group1_C10_instantPI_1()
prim_axon_1, prim_soma1_1 = Group1_C10_phas1()

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
                'prims': [{'axon': prim_axon_0, 'soma1': prim_soma1, 'router': Router_Prim_1, 'soma2': None}],
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
                'prims': [{'axon': prim_axon_1, 'soma1': prim_soma1_1, 'router': None, 'soma2': None}],
                'instant_prims': [{'axon': None, 'soma1': None, 'router': Router_Prim_3, 'soma2': None}],
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
        test_phase.append((group_id, i+1))  # 确保每个group的所有core执行相同的phase数
# 测试配置
test_config = {
    'tb_name': 'I11001',
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
