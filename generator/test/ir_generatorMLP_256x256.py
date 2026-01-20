# coding: utf-8
import numpy as np
import copy

np.random.seed(0x1000)

from primitive import Prim_02_Axon
from primitive import Prim_03_Axon
from primitive import Prim_04_Axon
from primitive import Prim_X5_Soma
from primitive import Prim_06_move_merge
from primitive import Prim_06_move_split
from primitive import Prim_07_LUT
# from primitive import Prim_08_lif
from primitive import Prim_09_Router
from primitive import Prim_41_Axon
from primitive import Prim_43_Axon
from primitive import Prim_81_Axon
from primitive import Prim_83_Axon


# from .prim_tool import *


class LogicalIRGenerator(object):
    def __init__(self):
        super().__init__()
        self.prim_list0 = []
        self.prim_list1 = []
        self.prim_list1_1 = []
        self.prim_list2 = []
        self.prim_list2_1 = []
        self.prim_list3 = []
        self.prim_list3_1 = []
        self.prim_list4 = []
        self.prim_list4_1 = []
        self.prim_list5 = []
        self.group0_phase_num = 3
        self.group1_phase_num = 3
        self.sending_phase_num = 1
        # self.receiveing_phase_num = 1
        # self.prim_list = []

    def format_ir_group1(self):  # IR for group1
        group0_config = {}
        # group_config_temp = {}
        group0_config['clock'] = 4000
        group0_config['mode'] = 1
        # core1_config = self.config_core1()
        group0_config[((0, 0), (1, 0))] = self.config_core1_0()
        group0_config[((0, 0), (2, 0))] = self.config_core2_0()
        group0_config[((0, 0), (2, 1))] = self.config_core2_1()
        group0_config[((0, 0), (1, 1))] = self.config_core1_1()
        # group_config[((0, 0), (5, 0))] = core1_0_config

        group1_config = {}
        group1_config['clock'] = 4000
        group1_config['mode'] = 1
        # core0_1_config = self.config_core0_1()
        group1_config[((0, 0), (0, 1))] = self.config_core0_1()
        # group1_config[((0, 0), (0, 2))] = self.config_core0_2()

        group2_config = {}
        group2_config['clock'] = 4000
        group2_config['mode'] = 1
        group2_config[((0, 0), (0, 0))] = self.config_core0_0()

        map_config = {
            'sim_clock': 11000,  # *2,
            'step_clock': {
                ((0, 0), 0): (11999, 12000),  # (0,0)-chip坐标，0表示trig:(可设置0~3)，20表示chock0_in_step, 50表示chock1_in_step.
            },
            0: {  # step group id
                # 'clock': None,
                0: group0_config,  # chip id 暂时还是（0，0）
                1: group1_config,  # chip id 暂时还是（0，0）
                2: group2_config,
            },
            # 1:{  # step group id
            #     # 'clock': None,
            #     0: group_config0,  # chip id 暂时还是（0，0）
            #     1: group_config,   # chip id 暂时还是（0，0）
        }
        return map_config

    def generate_logical_mapping_ir(self, neural_network=0, strategy=0):
        return self.format_ir_group1()

    def sending_prim(self):
        for phase in range(self.sending_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                # 0x09 router
                prim_router = Prim_09_Router()
                prim_router.Rhead_mode = 1
                prim_router.CXY = 0b00
                prim_router.Send_en = 1
                prim_router.Receive_en = 0
                prim_router.Dout_Mem_sel = 0
                prim_router.Send_number = 63  # 发送的所有包数，包含EN=0的包，需要减1
                prim_router.Addr_Dout_base = 0x400  # 4B寻址 24000
                prim_router.Addr_Dout_length = 31  # 16B的个数，需要减1
                prim_router.Addr_Rhead_base = 0x300  # 4B寻址 20C00
                prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
                prim_router.Addr_Din_base = 0x800  # 4B寻址 21000
                prim_router.Addr_Din_length = 0  # 需要减1  224*3/8
                prim_router.Receive_number = 0
                prim_router.Nx = 0
                prim_router.Ny = 0
                prim_router.Send_PI_en = 1
                prim_router.Back_sign_en = 0
                prim_router.Send_PI_num = 0
                prim_router.Receive_sign_num = 0
                prim_router.Send_PI_addr_base = 0x780 >> 2
                prim_router.Relay_number = 0
                prim_router.Q = 0
                prim_router.T_mode = 1
                prim_router.Receive_sign_en = 1
                prim_router.Soma_in_en = 0

                # prim_router = prim_router.init_data()
                const = 4;
                prim0_in = []
                for i in range((prim_router.Addr_Dout_length + 1) * 4):
                    tmp = []
                    for j in range(4):
                        if i == 0 or i == 1:
                            tmp.append(const)
                        else:
                            tmp.append(np.random.randint(255))
                    prim0_in.append(tmp)

                prim_router.memory_blocks = [
                    {'name': 'Dout_data',
                     'start': prim_router.Addr_Dout_base + 0x8000,
                     'data': prim0_in,
                     'mode': 0}]

                prim_router.send_destin_core_grp = [
                    {'core_id': [((0, 0), (1, 0)), ((0, 0), (2, 0))], 'data_num': 32, 'T_mode': 1, 'Rhead_num': 1},
                    {'core_id': [((0, 0), (1, 1)), ((0, 0), (2, 1))], 'data_num': 32, 'T_mode': 1, 'Rhead_num': 1}]

                prim_router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0, pack_per_Rhead=31, A_offset=0, Const=31,
                                     EN=1)  # 一对一包头参数形式
                prim_router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=1, A=0, pack_per_Rhead=31, A_offset=0, Const=31,
                                     EN=1)  # 一对一包头参数形式

                prim_router.instant_prim_request = [(((0, 0), (1, 0)), 0)]

                prim_router.add_instant_pi(PI_addr_offset=0, A_valid=0, S1_valid=0, R_valid=1, S2_valid=1, X=1, Y=0,
                                           Q=1)

                # phase dict

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = None
                self.prim_list0.append(one_phase_dict)

    def config_core0_0(self):
        self.sending_prim()

        config_core0_0 = {  # core id     # length先不计算
            # 'memory_blocks': {
            #     #  phase1
            #     'Dout_data': {'start': 0x8400, 'length': 16, 'initialize': True},
            # },
            'prims': self.prim_list0

        }
        return config_core0_0

    def config_prim1_0(self):
        for phase in range(self.group0_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                prim_soma1 = self.prim_RS2_G0_phase0(batch=phase + 1)

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list1.append(one_phase_dict)

            elif (phase + 1) == 2:
                one_phase_dict = {}
                prim_router, prim_soma2 = self.prim_RS2_G0_phase1(batch=phase + 1, CXY=1, Nx=1, Ny=0, Back_sign_en=1,
                                                                  Relay_number=31)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = prim_soma2
                self.prim_list1_1.append(one_phase_dict)

            elif (phase + 1) == 3:
                one_phase_dict = {}
                prim_axon, prim_soma1, prim_router = self.prim_AS1R(batch=phase + 1, x=-1, y=1, Back_sign_en=1,
                                                                    A=0)  # 11500

                # phase dict
                one_phase_dict['axon'] = prim_axon
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = None
                self.prim_list1_1.append(one_phase_dict)

    def config_core1_0(self):
        self.config_prim1_0()

        config_core1_0 = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase0
                'Router_receive_batch1': {'start': 0x0000, 'length': 256, 'initialize': True},
                #  phase1
                'Router_receive_batch2': {'start': 0x8400, 'length': 256, 'initialize': False},
                #  phase2
                'In_A_04_batch3': {'start': 0x4000, 'length': 256, 'initialize': False},
                'In_B_04_batch3': {'start': 0x0000, 'length': 64 * 1024, 'initialize': True},

                'input_x1_batch3': {'start': 0x7000, 'length': 16, 'initialize': False},
                'LUT_batch3': {'start': 0x7400, 'length': 256, 'initialize': True},

            },
            'prims': self.prim_list1,
            'instant_prims': self.prim_list1_1,
            'registers': {
                "Receive_PI_addr_base": 0x7a0 >> 2,
                "PI_CXY": 1,
                "PI_Nx": 1,
                "PI_Ny": 0,
                "PI_sign_CXY": 0,
                "PI_sign_Nx": 0,
                "PI_sign_Ny": 0,
                "instant_PI_en": 1,
                "fixed_instant_PI": 1,
                "instant_PI_number": 1,
                "PI_loop_en": 0,
                "start_instant_PI_num": 0}
        }
        return config_core1_0

    def config_prim2_0(self):
        for phase in range(self.group0_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                prim_soma1 = self.prim_RS2_G0_phase0(batch=phase + 1)

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list2.append(one_phase_dict)

            elif (phase + 1) == 2:
                one_phase_dict = {}
                prim_router, prim_soma2 = self.prim_RS2_G0_phase1(batch=phase + 1, CXY=0, Nx=0, Ny=0, Back_sign_en=0,
                                                                  Relay_number=0)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = prim_soma2
                self.prim_list2_1.append(one_phase_dict)

            elif (phase + 1) == 3:
                one_phase_dict = {}
                prim_axon, prim_soma1, prim_router = self.prim_AS1R(batch=phase + 1, x=-2, y=1, Back_sign_en=0,
                                                                    A=32)  # 11500

                # phase dict
                one_phase_dict['axon'] = prim_axon
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = None
                self.prim_list2_1.append(one_phase_dict)

    def config_core2_0(self):
        self.config_prim2_0()

        config_core2_0 = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase0
                'Router_receive_batch1': {'start': 0x0000, 'length': 256, 'initialize': True},
                #  phase1
                'Router_receive_batch2': {'start': 0x8400, 'length': 256, 'initialize': False},
                #  phase2
                'In_A_04_batch3': {'start': 0x4000, 'length': 256, 'initialize': False},
                'In_B_04_batch3': {'start': 0x0000, 'length': 64 * 1024, 'initialize': True},

                'input_x1_batch3': {'start': 0x7000, 'length': 16, 'initialize': False},
                'LUT_batch3': {'start': 0x7400, 'length': 256, 'initialize': True},

            },
            'prims': self.prim_list2,
            'instant_prims': self.prim_list2_1,
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
                "instant_PI_number": 1,
                "PI_loop_en": 0,
                "start_instant_PI_num": 0}
        }
        return config_core2_0

    def config_prim2_1(self):
        for phase in range(self.group0_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                prim_soma1 = self.prim_RS2_G0_phase0(batch=phase + 1)

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list3.append(one_phase_dict)

            elif (phase + 1) == 2:
                one_phase_dict = {}
                prim_router, prim_soma2 = self.prim_RS2_G0_phase1(batch=phase + 1, CXY=0, Nx=0, Ny=0, Back_sign_en=0,
                                                                  Relay_number=0)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = prim_soma2
                self.prim_list3_1.append(one_phase_dict)

            elif (phase + 1) == 3:
                one_phase_dict = {}
                prim_axon, prim_soma1, prim_router = self.prim_AS1R(batch=phase + 1, x=-2, y=0, Back_sign_en=0,
                                                                    A=96)  # 11500

                # phase dict
                one_phase_dict['axon'] = prim_axon
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = None
                self.prim_list3_1.append(one_phase_dict)

    def config_core2_1(self):
        self.config_prim2_1()

        config_core2_1 = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase0
                'Router_receive_batch1': {'start': 0x0000, 'length': 256, 'initialize': True},
                #  phase1
                'Router_receive_batch2': {'start': 0x8400, 'length': 256, 'initialize': False},
                #  phase2
                'In_A_04_batch3': {'start': 0x4000, 'length': 256, 'initialize': False},
                'In_B_04_batch3': {'start': 0x0000, 'length': 64 * 1024, 'initialize': True},

                'input_x1_batch3': {'start': 0x7000, 'length': 16, 'initialize': False},
                'LUT_batch3': {'start': 0x7400, 'length': 256, 'initialize': True},
            },
            'prims': self.prim_list3,
            'instant_prims': self.prim_list3_1,
            'registers': {
                "Receive_PI_addr_base": 0x7a0 >> 2,
                "PI_CXY": 1,
                "PI_Nx": -1,
                "PI_Ny": 0,
                "PI_sign_CXY": 0,
                "PI_sign_Nx": 0,
                "PI_sign_Ny": 0,
                "instant_PI_en": 1,
                "fixed_instant_PI": 1,
                "instant_PI_number": 1,
                "PI_loop_en": 0,
                "start_instant_PI_num": 0}
        }
        return config_core2_1

    def config_prim1_1(self):
        for phase in range(self.group0_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                prim_soma1 = self.prim_RS2_G0_phase0(batch=phase + 1)

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list4.append(one_phase_dict)

            elif (phase + 1) == 2:
                one_phase_dict = {}
                prim_router, prim_soma2 = self.prim_RS2_G0_phase1(batch=phase + 1, CXY=1, Nx=1, Ny=0, Back_sign_en=0,
                                                                  Relay_number=31)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = prim_soma2
                self.prim_list4_1.append(one_phase_dict)

            elif (phase + 1) == 3:
                one_phase_dict = {}
                prim_axon, prim_soma1, prim_router = self.prim_AS1R(batch=phase + 1, x=-1, y=0, Back_sign_en=0,
                                                                    A=64)  # 11500

                # phase dict
                one_phase_dict['axon'] = prim_axon
                one_phase_dict['soma1'] = prim_soma1
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = None
                self.prim_list4_1.append(one_phase_dict)

    def config_core1_1(self):
        self.config_prim1_1()

        config_core1_1 = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase0
                'Router_receive_batch1': {'start': 0x0000, 'length': 256, 'initialize': True},
                #  phase1
                'Router_receive_batch2': {'start': 0x8400, 'length': 256, 'initialize': False},
                #  phase2
                'In_A_04_batch3': {'start': 0x4000, 'length': 256, 'initialize': False},
                'In_B_04_batch3': {'start': 0x0000, 'length': 64 * 1024, 'initialize': True},

                'input_x1_batch3': {'start': 0x7000, 'length': 16, 'initialize': False},
                'LUT_batch3': {'start': 0x7400, 'length': 256, 'initialize': True},
            },
            'prims': self.prim_list4,
            'instant_prims': self.prim_list4_1,
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
                "instant_PI_number": 1,
                "PI_loop_en": 0,
                "start_instant_PI_num": 0}
        }
        return config_core1_1

    def config_prim0_1(self):
        for phase in range(self.group1_phase_num):
            if (phase + 1) == 1:
                one_phase_dict = {}
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list5.append(one_phase_dict)

            elif (phase + 1) == 2:
                one_phase_dict = {}
                prim_router = Prim_09_Router()
                prim_router.Rhead_mode = 1
                prim_router.CXY = 0
                prim_router.Send_en = 0
                prim_router.Receive_en = 1
                prim_router.Dout_Mem_sel = 0
                prim_router.Send_number = 0  # 发送的所有包数，包含EN=0的包，需要减1
                prim_router.Addr_Dout_base = 0  # 4B寻址
                prim_router.Addr_Dout_length = 0  # 16B的个数，需要减1
                prim_router.Addr_Rhead_base = 0x300  # 4B寻址
                prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
                prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
                prim_router.Addr_Din_length = 127  # 需要减1  224*16/8
                prim_router.Receive_number = 3
                prim_router.Nx = 0
                prim_router.Ny = 0
                prim_router.Send_PI_en = 1
                prim_router.Back_sign_en = 0
                prim_router.Send_PI_num = 0
                prim_router.Receive_sign_num = 0
                prim_router.Send_PI_addr_base = 0x780 >> 2
                prim_router.Relay_number = 0
                prim_router.Q = 0
                prim_router.T_mode = 1
                prim_router.Receive_sign_en = 1
                prim_router.Soma_in_en = 0
                # prim_in = prim_router.init_data()
                # prim_router.memory_blocks = [
                #     {'name': 'Router_receive',
                #         'start': 0x8400,
                #         'length': 3584,  #224*16
                #         'data':  prim_in[0],
                #         'mode': 0},
                # ]

                prim_router.recv_source_core_grp = [
                    {"core_id": [((0, 0), (1, 0))], "data_num": 32, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": [((0, 0), (2, 0))], "data_num": 32, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": [((0, 0), (2, 1))], "data_num": 32, "T_mode": 1, "Rhead_num": 1},
                    {"core_id": [((0, 0), (1, 1))], "data_num": 32, "T_mode": 1, "Rhead_num": 1}]

                prim_router.instant_prim_request = [(((0, 0), (1, 0)), 0)]
                prim_router.add_instant_pi(PI_addr_offset=2, A_valid=1, S1_valid=1, R_valid=1, S2_valid=0, X=1, Y=-1,
                                           Q=1)  # pi_offset按顺序执行

                prim_soma2 = Prim_06_move_merge()
                prim_soma2.length_in = 1024
                prim_soma2.length_out = 1024
                prim_soma2.length_ciso = 1
                prim_soma2.num_in = 1
                prim_soma2.num_out = 1
                prim_soma2.num_ciso = 1
                prim_soma2.type_in = 1
                prim_soma2.type_out = 1
                prim_soma2.in_cut_start = 0
                prim_soma2.Reset_Addr_in = 1
                prim_soma2.Reset_Addr_out = 1
                prim_soma2.Reset_Addr_ciso = 1
                prim_soma2.Row_ck_on = 0
                prim_soma2.Addr_Start_in = 0x8400  # 21000
                prim_soma2.Addr_Start_out = 0x0000
                prim_soma2.Addr_Start_ciso = 0x89e0  # 1E6A0
                prim_soma2.in_row_max = 0
                prim2_in = prim_soma2.init_data()
                prim_soma2.memory_blocks = [
                    {'name': 'Router_receive_batch{}',
                     'start': 0x8400,
                     # 'length': 3584,    #  224*16
                     'data': prim2_in[0],
                     'mode': 0,
                     'initialize': False},
                    # {'name': 'Input_buff_RGB',
                    #     'start': start_out,
                    #     'length': 3584,
                    #     'data': prim2_in[1],
                    #     'mode': 0},
                ]
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = prim_router
                one_phase_dict['soma2'] = prim_soma2
                self.prim_list5.append(one_phase_dict)

            elif (phase + 1) == 3:
                one_phase_dict = {}
                Axon_Prim_02 = Prim_02_Axon()
                Axon_Prim_02.InA_type = 0  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
                Axon_Prim_02.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
                Axon_Prim_02.Bias_length = 0
                Axon_Prim_02.pad_on = False
                Axon_Prim_02.avg_pooling_en = False
                Axon_Prim_02.Input_fm_Px = 1
                Axon_Prim_02.Input_fm_Py = 1
                Axon_Prim_02.pooling_Kx = 1
                Axon_Prim_02.pooling_Ky = 2
                Axon_Prim_02.pooling_Sx = 1
                Axon_Prim_02.pooling_Sy = 1
                Axon_Prim_02.pad_top = 0
                Axon_Prim_02.pad_down = 0
                Axon_Prim_02.pad_left = 0
                Axon_Prim_02.pad_right = 0
                Axon_Prim_02.cin = 256
                Axon_Prim_02.cout = 256
                Axon_Prim_02.Reset_Addr_A = 1
                Axon_Prim_02.Reset_Addr_V = 1
                Axon_Prim_02.Addr_InA_base = 0x0000
                Axon_Prim_02.Addr_Bias_base = 0x0080
                Axon_Prim_02.Addr_V_base = 0x4000
                Axon_Prim_02.constant_b = 0
                Axon_Prim_02.A2S2_mode = 0

                prim_in = Axon_Prim_02.init_data()
                blocks = [{'name': "P02_input_X_batch{}",
                           'start': Axon_Prim_02.Addr_InA_base,
                           'data': prim_in[0],
                           'mode': 0,
                           'initialize': False}]
                # if Axon_Prim_02.Load_Bias == 2 or Axon_Prim_02.Load_Bias == 3:
                #     blocks.append({'name': "P02_bias",
                #                    'start': Axon_Prim_02.Addr_Bias_base,
                #                    'data': prim_in[1],
                #                    'mode': 0})
                # phase dict
                one_phase_dict['axon'] = Axon_Prim_02
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list5.append(one_phase_dict)

    def config_core0_1(self):
        self.config_prim0_1()

        core0_1_cofig = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase2
                'phase2_XW_buff': {'start': 0x8400, 'length': 1024, 'initialize': False},
                'phase2_ciso': {'start': 0x89e0, 'length': 1, 'initialize': True},
                'phase4_P02_input_X': {'start': 0x0000, 'length': 512, 'initialize': False}
            },
            'prims': self.prim_list5
        }
        return core0_1_cofig

    def prim_RS2_G0_phase1(self, batch, CXY, Nx, Ny, Back_sign_en, Relay_number):
        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = CXY
        prim_router.Send_en = 0
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 0
        prim_router.Send_number = 0  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0  # 4B寻址
        prim_router.Addr_Dout_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = 0  # 4B寻址
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 31  # 需要减1  56*4*32
        prim_router.Receive_number = 0
        prim_router.Nx = Nx
        prim_router.Ny = Ny
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = Back_sign_en
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = Relay_number
        prim_router.Q = 0
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 0

        prim_router.recv_source_core_grp = [{'core_id': ((0, 0), (0, 0)), 'data_num': 32, 'T_mode': 1, 'Rhead_num': 1}]

        prim_router.instant_request_back = [((0, 0), (0, 0))]
        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_receive',
        #         'start': 0x8400,
        #         'length': 3584,  #224*16
        #         'data':  prim_in[0],
        #         'mode': 0},
        # ]

        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 256
        prim_soma2.length_out = 256
        prim_soma2.length_ciso = 1
        prim_soma2.num_in = 1
        prim_soma2.num_out = 1
        prim_soma2.num_ciso = 1
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400  # 21000
        prim_soma2.Addr_Start_out = 0x4000
        prim_soma2.Addr_Start_ciso = 0x89e0  # 1E6A0
        prim_soma2.in_row_max = 0
        prim2_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Router_receive_batch{}'.format(batch),
             'start': 0x8400,
             'data': prim2_in[0],
             'mode': 0,
             'initialize': False},
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]
        return prim_router, prim_soma2

    def prim_AS1R(self, batch, x, y, Back_sign_en, A):
        prim_axon = Prim_04_Axon()
        prim_axon.InA_type = 1
        prim_axon.InB_type = 1
        prim_axon.Load_Bias = 0
        prim_axon.cin = 256
        prim_axon.cout = 256
        prim_axon.Reset_Addr_A = 1
        prim_axon.Addr_InA_base = 0x4000
        prim_axon.Addr_InB_base = 0x0000
        prim_axon.Addr_Bias_base = 0x7e80
        prim_axon.Addr_V_base = 0x7000  # 流水计算
        prim_axon.A2S2_mode = 0

        prim_in = prim_axon.init_data()
        prim_axon.constant_b = 0
        prim_axon.memory_blocks = [
            {'name': 'In_A_04_batch{}'.format(batch),
             'start': prim_axon.Addr_InA_base,
             # 'length': 2048,  # 64*32
             'data': prim_in[0],
             'mode': 0,
             'initialize': False},
            {'name': 'In_B_04_batch{}'.format(batch),
             'start': prim_axon.Addr_InB_base,
             # 'length': 2048,
             'data': prim_in[1],
             'mode': 0,
             'initialize': True},
        ]

        Soma1_prim_07 = Prim_07_LUT()
        Soma1_prim_07.neuron_real_num = 32
        Soma1_prim_07.group_num = 8
        Soma1_prim_07.reset_Addr_X = 1
        Soma1_prim_07.reset_Addr_Y = 1
        Soma1_prim_07.Row_ck_on = 1
        Soma1_prim_07.Addr_X_Start = prim_axon.Addr_V_base  # 流水地址
        Soma1_prim_07.X_type = 0  # 0:int32 1/2/3 int8
        Soma1_prim_07.Addr_Start_out = 0x9000
        Soma1_prim_07.Y_type = 1
        Soma1_prim_07.Addr_LUT_Start = 0x7400  # 这里需要设置一个合适的地址
        Soma1_prim_07.LUT_DW = 2  # 0:4b   1:8b    2:12b   3:16b
        Soma1_prim_07.X_cut_start = 3
        Soma1_prim_07.in_row_max = 2
        Soma1_prim_07.mem_sel = 1

        prim_in = Soma1_prim_07.init_data()
        Soma1_prim_07.memory_blocks = [
            {'name': 'input_x1_batch{}'.format(batch),
             'start': Soma1_prim_07.Addr_X_Start,
             'data': prim_in[0],
             'mode': 0,
             'initialize': False},
            {'name': 'LUT_batch{}'.format(batch),
             'start': Soma1_prim_07.Addr_LUT_Start,
             'data': prim_in[1],
             'mode': 0,
             'initialize': True}
        ]

        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0
        prim_router.Send_en = 1
        prim_router.Receive_en = 0
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 31  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 24000
        prim_router.Addr_Dout_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = 0x300  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 0  # 需要减1  224*3/8
        prim_router.Receive_number = 0
        prim_router.Nx = 0
        prim_router.Ny = 0
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = Back_sign_en
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0x8f00 >> 2
        prim_router.Relay_number = 0
        prim_router.Q = 0
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        prim_router.send_destin_core_grp = [{'core_id': ((0, 0), (0, 1)), 'data_num': 32, 'T_mode': 1, 'Rhead_num': 1}]
        prim_router.instant_request_back = [((0, 0), (0, 1))]

        prim_router.addRHead(
            S=0, T=1, P=0, Q=0, X=x, Y=y, A=A, pack_per_Rhead=31, A_offset=0, Const=31, EN=1)  # 一对一包头参数形式

        return prim_axon, Soma1_prim_07, prim_router

    def prim_RS2_G0_phase0(self, batch):
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 256
        prim_soma1.length_out = 256
        prim_soma1.length_ciso = 1
        prim_soma1.num_in = 1
        prim_soma1.num_out = 1
        prim_soma1.num_ciso = 1
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = 0x0000  # 21000
        prim_soma1.Addr_Start_out = 0x4000
        prim_soma1.Addr_Start_ciso = 0x89e0  # 1E6A0
        prim_soma1.in_row_max = 0
        prim1_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Router_receive_batch{}'.format(batch),
             'start': 0x0000,
             'data': prim1_in[0],
             'mode': 0,
             'initialize': True},
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]
        return prim_soma1

xxx = LogicalIRGenerator()
yyy = xxx.generate_logical_mapping_ir()

from generator.test.get_router_info import get_router_info

router_static = get_router_info(yyy)

import scipy.io as scio

scio.savemat('./chip_2_1_LUT.mat', {'router_static': router_static})
