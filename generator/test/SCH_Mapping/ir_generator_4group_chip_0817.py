'''
Simulated sending chip(3cores) - 3 sending cores - Group1
[3 sending cores - Group1] is 1 group 
'''
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
from primitive import Prim_08_lif
from primitive import Prim_09_Router
from primitive import Prim_41_Axon
from primitive import Prim_43_Axon
from primitive import Prim_81_Axon
from primitive import Prim_83_Axon

# from .prim_tool import *



class LogicalIRGenerator(object):
    def __init__(self):
        super().__init__()
        self.prim_list = []
        self.prim_list_extra = []
        self.prim_list_g2 = []
        self.prim_list_c1 = []
        self.prim_list_c2 = []
        self.prim_list_c3 = []
        self.IB0_prim_list_c1 = []
        self.IB0_prim_list_c2 = []
        self.IB0_prim_list_c3 = []
        self.group1_phase_num = 19
        self.group2_phase_num = 22   # 1
        self.sending_phase_num = 19
        self.L50_phase_num = 3
        self.sync_num = 24
        self.addd = 0
        # self.prim_list = []

    def format_ir_group1(self):   #  IR for group1
        # Group1
        group_config = {}
        # group_config_temp = {}
        group_config['clock'] = 35000
        group_config['mode'] = 1
        core1_config = self.config_core1()
        core1_extra = self.config_core1_extra()
        # group_config[((1, 0), (0, 0))] = core1_config
        group_config[((1, 0), (14, 0))] = copy.deepcopy(core1_extra)
        group_config[((1, 0), (15, 0))] = copy.deepcopy(core1_extra)
        group_config[((1, 0), (14, 1))] = copy.deepcopy(core1_extra)
        group_config[((1, 0), (15, 1))] = copy.deepcopy(core1_extra)

        # (15, 0) 第一个phase收数据
        router = Prim_09_Router()
        router.Rhead_mode = 1
        router.CXY = 0
        router.Send_en = 0
        router.Receive_en = 1
        router.Addr_Dout_base = 0
        router.Dout_Mem_sel = 0
        router.Addr_Dout_length = 0
        router.Send_number = 0
        router.Addr_Rhead_base = 0
        router.Addr_Rhead_length = 0
        router.Addr_Din_base = 0x400
        router.Addr_Din_length = 255
        router.Receive_number = 15
        router.Nx = 0
        router.Ny = 0
        router.Send_PI_en = 0
        router.Back_sign_en = 0
        router.Send_PI_num = 0
        router.Receive_sign_num = 0
        router.Send_PI_addr_base = 0
        router.Relay_number = 0
        router.Q = 0
        router.Recevie_sign_en = 0
        router.T_mode = 1
        router.Soma_in_en = 0
        for ss in range(16):
            router.recv_source_core_grp.append({'core_id': [((2, 0), (ss, 4))],
                                                'data_num': 128,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': 0
                                                })
        group_config[((1, 0), (15, 0))]['prims'][0]['router'] = router

        for i in range(14):     #  14
            for j in range(2):    #  2
                # if (i, j) == (0, 0):
                #     continue
                group_config[((1, 0), (i, j))] = self.config_core_from_core1(
                    core1_config, i, j)

        # DB CORE: sending chip group
        group_config0 = {}
        group_config0['clock'] = 35000
        group_config0['mode'] = 1
        group_config0[((0, 0), (15, 1))] = self.IB0_config_sending_c1()
        group_config0[((0, 0), (15, 2))] = self.IB0_config_sending_c2()
        group_config0[((0, 0), (15, 3))] = self.IB0_config_sending_c3()
        # IB CORE: 3sending cores
        group_config00 = {}
        group_config00['clock'] = 35000
        group_config00['mode'] = 1
        group_config00[((1, 0), (0, 2))] = self.config_sending_c1()
        group_config00[((1, 0), (0, 3))] = self.config_sending_c2()
        group_config00[((1, 0), (0, 4))] = self.config_sending_c3()
        # Group 2
        group_config2 = {}
        group_config2['clock'] = 35000
        group_config2['mode'] = 1
        G2core1_config = self.G2_config_core1()
        group_config2[((1, 0), (2, 2))] = G2core1_config
        for i in range(2,16):     #  2-16
            for j in range(2,5):    #  2-5
                if (i, j) == (2, 2):
                    continue
                group_config2[((1, 0), (i, j))] = self.G2config_core_from_core1(
                    G2core1_config, i, j)
                              
        map_config = {
            'sim_clock': 35000*24,   #  *2,
            # 'step_clock': {
            #     ((0, 0), 0): (10000*24-1, 10000*24),  #  (0,0)-chip坐标，0表示trig:(可设置0~3)，20表示chock0_in_step, 50表示chock1_in_step.
            #     ((1, 0), 0): (10000*24-1, 10000*24),
            #     },
            ((0, 0), 0): {  # step group id
                'step_exe_number':1,
                0: group_config0,  # chip id 暂时还是（0，0）
            },
            ((1, 0), 0): {  # step group id
                'step_exe_number':1,
                1: group_config00,  
                2: group_config, 
                3: group_config2,   
            },
        }

        return map_config


    def generate_logical_mapping_ir(self, neural_network, strategy):

        return self.format_ir_group1()  #, self.format_ir_source()  

    def IB0_prim_sending_c1(self):
        for phase in range(self.sync_num):  
            if (phase+1) == 1:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1, y=1, Addr_Rhead_base=0x300, idx=0, idy=2, phase=phase, move_in=0x4000)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 2:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x308, idx=0, idy=2, phase=phase, move_in=0x4700)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 3:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x310, idx=0, idy=2, phase=phase, move_in=0x4E00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict)    

            elif (phase+1) == 4:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x318, idx=0, idy=2, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict) 

            elif (phase+1) == 5:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x320, idx=0, idy=2, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict) 

            elif (phase+1) == 6:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x328, idx=0, idy=2, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 7:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x330, idx=0, idy=2, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c1.append(one_phase_dict)

            else:
                self.IB0_prim_list_c1.append(self.none_phase())

    def IB0_prim_sending_c2(self):
        for phase in range(self.sync_num):  
            if (phase+1) == 8:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x300, idx=0, idy=3, phase=phase, move_in=0x4000)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 9:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x308, idx=0, idy=3, phase=phase, move_in=0x4700)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 10:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1, x=1,y=1, Addr_Rhead_base=0x310, idx=0, idy=3, phase=phase, move_in=0x4E00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict)    

            elif (phase+1) == 11:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x318, idx=0, idy=3, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict) 

            elif (phase+1) == 12:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x320, idx=0, idy=3, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict) 

            elif (phase+1) == 13:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x328, idx=0, idy=3, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 14:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x330, idx=0, idy=3, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c2.append(one_phase_dict)

            else:
                self.IB0_prim_list_c2.append(self.none_phase())

    def IB0_prim_sending_c3(self):
        for phase in range(self.sync_num):  
            if (phase+1) == 15:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x300, idx=0, idy=4, phase=phase, move_in=0x4000)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 16:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x308, idx=0, idy=4, phase=phase, move_in=0x4700)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 17:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x310, idx=0, idy=4, phase=phase, move_in=0x4E00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict)    

            elif (phase+1) == 18:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x318, idx=0, idy=4, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict) 

            elif (phase+1) == 19:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x320, idx=0, idy=4, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict) 

            elif (phase+1) == 20:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x328, idx=0, idy=4, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 21:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_s(batch=phase+1,x=1,y=1, Addr_Rhead_base=0x330, idx=0, idy=4, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma2]
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = None
                self.IB0_prim_list_c3.append(one_phase_dict)

            else:
                self.IB0_prim_list_c3.append(self.none_phase())

    def IB0_config_sending_c1(self):  # sendding core1
        self.IB0_prim_sending_c1()
        sending1_cofig = {  # core id
            'memory_blocks': {
                'IR0_sending_batch1': {'start': 0x4000, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch2': {'start': 0x4700, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch3': {'start': 0x4E00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch4': {'start': 0x5500, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch5': {'start': 0x5C00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch6': {'start': 0x6300, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch7': {'start': 0x6A00, 'length': 224*16*2, 'initialize': True}, 
            },
            'prims': self.IB0_prim_list_c1
        }
        return sending1_cofig

    def IB0_config_sending_c2(self):  # sendding core1
        self.IB0_prim_sending_c2()
        sending2_cofig = {  # core id
            'memory_blocks': {
                'IR0_sending_batch8': {'start': 0x4000, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch9': {'start': 0x4700, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch10': {'start': 0x4E00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch11': {'start': 0x5500, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch12': {'start': 0x5C00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch13': {'start': 0x6300, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch14': {'start': 0x6A00, 'length': 224*16*2, 'initialize': True}, 
            },
            'prims': self.IB0_prim_list_c2
        }
        return sending2_cofig

    def IB0_config_sending_c3(self):  # sendding core1
        self.IB0_prim_sending_c3()
        sending3_cofig = {  # core id
            'memory_blocks': {
                'IR0_sending_batch15': {'start': 0x4000, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch16': {'start': 0x4700, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch17': {'start': 0x4E00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch18': {'start': 0x5500, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch19': {'start': 0x5C00, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch20': {'start': 0x6300, 'length': 224*16*2, 'initialize': True},
                'IR0_sending_batch21': {'start': 0x6A00, 'length': 224*16*2, 'initialize': True}, 
            },
            'prims': self.IB0_prim_list_c3
        }
        return sending3_cofig

    def prim_router_s(self,batch, x, y, Addr_Rhead_base, idx, idy, phase, move_in):
        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 224
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 32
        prim_soma2.num_ciso = 32
        prim_soma2.length_out = 224
        prim_soma2.num_out = 32
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = move_in   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma2.Addr_Start_out = 0x9000   # 0x4000  # 10000
        prim_soma2.in_row_max = 0
        prim_soma2.mem_sel = 1
        prim_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'IR0_Sending_batch{}'.format(batch),
                'start': prim_soma2.Addr_Start_in,
                # 'length': 672,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': True},
        ]

        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 0
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 895  # 224*32/8
              
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 21000
        prim_router.Addr_Dout_length = 447  # 16B的个数，需要减1 224*32/16
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 1  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 0  # 需要减1 224*32/8
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1
        
        prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx, idy))], 'data_num':896,'T_mode': 1, 'Rhead_num': 1, 'sync_en':1, 'sync_phase_num': phase})
        prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=896-1, A_offset=0, Const=896-1, EN=1)  # 一对一包头参数形式
       
        return prim_router, prim_soma2

    # prim setting in 1-4 phases of core1
    def prim_RS2_phase1_4(self, start_out, Addr_Rhead_base, idx, idy, phase):
        # 0x09 rouer        
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b01
        prim_router.Send_en = 0
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 0
        prim_router.Send_number = 0   # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0   #  4B寻址 
        prim_router.Addr_Dout_length = 0   # 16B的个数，需要减1 
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1 
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000 
        prim_router.Addr_Din_length = 447   # 需要减1  224*16/8 
        prim_router.Receive_number = 0
        prim_router.Nx = 0
        prim_router.Ny = 1
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = 0
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = 447
        prim_router.Q = 0
        prim_router.T_mode = 0
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 0

        #  prim_router.send_destin_core_grp.append({'core_id': (1, 1), 'data_num': 84,'T_mode': 1, 'Rhead_num': 1})
        prim_router.recv_source_core_grp.append({'core_id': ((1,0),(idx, idy)), 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

        # prim_router.recv_source_core_grp[
        #     {'core_id': (0, 0), 'data_num': 10,'T_mode': 0, 'Rhead_num': 1}
        #     ]
            
        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_receive',
        #         'start': 0x8400,
        #         'length': 3584,  #224*16
        #         'data':  prim_in[0],
        #         'mode': 0},
        # ]

        #  move prim in 1-4 phases of core1
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 14*16
        prim_soma2.length_out = 14*16
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 16
        prim_soma2.num_out = 16
        prim_soma2.num_ciso = 0
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0 
        prim_soma2.Addr_Start_in = 0x8400     # 21000 
        prim_soma2.Addr_Start_out = start_out     
        prim_soma2.Addr_Start_ciso = 0x79A8   # 1E6A0
        prim_soma2.in_row_max = 0 
        prim2_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Router_receive_Phase1_4',
                'start': 0x8400,
                # 'length': 3584,    #  224*16
                'data':  prim2_in[0],
                'mode': 0,
                'initialize': False
                },
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]

        return prim_router, prim_soma2

    def prim_L1L2_arrange(self, start_in, start_out, Addr_Rhead_base, idx, idy, idxs, idys, x, y, phase, Ai, B=64):
        # 0x06 move 
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 56*32
        prim_soma1.length_ciso = 0
        prim_soma1.num_in = 4
        prim_soma1.num_ciso = 4
        prim_soma1.length_out = 56*32
        prim_soma1.num_out = 4
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = start_in    #  0x4380   
        prim_soma1.Addr_Start_ciso = 0x79A8  #  1E6A0
        prim_soma1.Addr_Start_out = 0x9000  #  30000
        prim_soma1.in_row_max = 0
        prim_soma1.mem_sel = 1 
        prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Sending_L2',
                'start': prim_soma1.Addr_Start_in,
                # 'length': 672,   #   流水（不一致的时候）的时候需要声明
                'data':  prim_in[0],
                'mode': 0,
                'initialize': True},
        ]

        #  0x09 rouer        
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b01
        prim_router.Send_en = 1
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 895   # 发送的所有包数，包含EN=0的包，需要减1  56*4*32/8   4*32/8
        prim_router.Addr_Dout_base = 0x1000   #  4B寻址 
        prim_router.Addr_Dout_length = 7   # 16B的个数，需要减1  4*32/16
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 
        prim_router.Addr_Rhead_length = 27  # 16B的个数，需要减1   56*8/16 ->  56*8/4=112
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000 
        prim_router.Addr_Din_length = 447   # 需要减1  224*16/8 
        prim_router.Receive_number = 0
        prim_router.Nx = 0
        prim_router.Ny = 1
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = 0
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = 447
        prim_router.Q = 0
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        x0 = x
        idxs0 = idxs

        for ii in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idxs, idys))], 'data_num': 16*4,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0

        for jj in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            #  prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num': 16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0
        
        for kk in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            #  prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num':16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0

        for mm in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            #  prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num': 16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1
          
        
        #  prim_router.send_destin_core_grp.append({'core_id': (1, 1), 'data_num': 84,'T_mode': 1, 'Rhead_num': 1})
        prim_router.recv_source_core_grp.append({'core_id': ((1,0),(idx, idy)), 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

        # prim_router.recv_source_core_grp[
        #     {'core_id': (0, 0), 'data_num': 10,'T_mode': 0, 'Rhead_num': 1}
        #     ]

        #  move prim in 1-4 phases of core1
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 14*16
        prim_soma2.length_out = 14*16
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 16
        prim_soma2.num_out = 16
        prim_soma2.num_ciso = 0
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0 
        prim_soma2.Addr_Start_in = 0x8400     # 21000 
        prim_soma2.Addr_Start_out = start_out     
        prim_soma2.Addr_Start_ciso = 0x79A8   # 1E6A0
        prim_soma2.in_row_max = 0 
        prim2_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Router_receive_Phase1_4',
                'start': 0x8400,
                # 'length': 3584,    #  224*16
                'data':  prim2_in[0],
                'mode': 0,
                'initialize': False
                },
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]

        return prim_soma1, prim_router, prim_soma2

    def prim_L1L2_arrange_s(self, start_in, Addr_Rhead_base, idx, idy, idxs, idys, x, y, phase, Ai, B=64):
        # 0x06 move 
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 56*32
        prim_soma1.length_ciso = 0
        prim_soma1.num_in = 4
        prim_soma1.num_ciso = 4
        prim_soma1.length_out = 56*32
        prim_soma1.num_out = 4
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = start_in    #  0x4380   
        prim_soma1.Addr_Start_ciso = 0x79A8  #  1E6A0
        prim_soma1.Addr_Start_out = 0x9000  #  30000
        prim_soma1.in_row_max = 0
        prim_soma1.mem_sel = 1 
        prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Sending_L2',
                'start': prim_soma1.Addr_Start_in,
                # 'length': 672,   #   流水（不一致的时候）的时候需要声明
                'data':  prim_in[0],
                'mode': 0,
                'initialize': True},
        ]

        #  0x09 rouer        
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 0
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 895   # 发送的所有包数，包含EN=0的包，需要减1  56*4*32/8   4*32/8
        prim_router.Addr_Dout_base = 0x1000   #  4B寻址 
        prim_router.Addr_Dout_length = 7   # 16B的个数，需要减1  4*32/16
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 
        prim_router.Addr_Rhead_length = 27  # 16B的个数，需要减1   56*8/16
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000 
        prim_router.Addr_Din_length = 0   # 需要减1  224*16/8 
        prim_router.Receive_number = 0
        prim_router.Nx = 0
        prim_router.Ny = 0
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = 0
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = 447
        prim_router.Q = 0
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        x0 = x
        idxs0 = idxs

        for ii in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idxs, idys))], 'data_num': 16*4,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0

        for jj in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            # prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num': 16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0
        
        for kk in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            # prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num':16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1

        Ai += 1
        x = x0
        idxs = idxs0

        for mm in range(14):
            prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=int(4*B*Ai/8), pack_per_Rhead=15, A_offset=int(32/8), Const=3, EN=1) # 一对一包头参数形式
            # prim_router.send_destin_core_grp.append({'core_id': [((0,0),(idxs, idys))], 'data_num': 16,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            x += 1
            idxs += 1
             
        return prim_soma1, prim_router


    def prim_top_overlap(self, s1_start_in, s2_start_out, x, y, color='R', Addr_Rhead_base=0x300):        # for  R, G, B
        # 0x06 move 
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 14*16
        prim_soma1.length_ciso = 0
        prim_soma1.num_in = 3
        prim_soma1.num_ciso = 3
        prim_soma1.length_out = 14*16
        prim_soma1.num_out = 3
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = s1_start_in      #  0x4380   # 10E00
        prim_soma1.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma1.Addr_Start_out = 0x9000  # 30000
        prim_soma1.in_row_max = 0
        prim_soma1.mem_sel = 1 
        prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Sending_top_overlap_{}'.format(color),
                'start': prim_soma1.Addr_Start_in,
                # 'length': 672,   #   流水（不一致的时候）的时候需要声明
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Sending_top_buff',
            #     'start': prim_soma1.Addr_Start_out,
            #     'length': 672,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 83  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 24000
        prim_router.Addr_Dout_length = 41  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 83  # 需要减1  224*3/8
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        prim_router.send_destin_core_grp.append({'core_id': ((1,0), (1, 0)), 'data_num': 84,'T_mode': 1, 'Rhead_num': 1})
        prim_router.recv_source_core_grp.append({'core_id': ((1,0), (13,0)), 'data_num': 84,'T_mode': 1, 'Rhead_num': 1})

        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_top_overlap_{}'.format(color),
        #         'start': prim_router.Addr_Dout_base,
        #         'data':  prim_in[0],
        #         'mode': 0,
        #         'initialize': False},
        # ]

        prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=0, pack_per_Rhead=83, A_offset=0, Const=83, EN=1) # 一对一包头参数形式
        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 14*16
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 3
        prim_soma2.num_ciso = 0
        prim_soma2.length_out = 14*16
        prim_soma2.num_out = 3
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma2.Addr_Start_out = s2_start_out    #  0x4000  # 10000
        prim_soma2.in_row_max = 0
        prim_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Receive_top_overlap_{}'.format(color),
                'start': prim_soma2.Addr_Start_in,
                # 'length': 672,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Receive_top_buff',
            #     'start': prim_soma2.Addr_Start_out,
            #     'length': 672,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        return prim_soma1, prim_router, prim_soma2 


    def prim_down_overlap(self, s1_start_in, s2_start_out, x, y, color='R', Addr_Rhead_base=0x300):             # for R,G,B
        # 0x06 move 
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 14*16
        prim_soma1.length_ciso = 0
        prim_soma1.num_in = 2
        prim_soma1.num_ciso = 2
        prim_soma1.length_out = 14*16
        prim_soma1.num_out = 2
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = s1_start_in      #   0x7A18   # 1E860
        prim_soma1.Addr_Start_ciso = 0x79A8 # 1E6A0
        prim_soma1.Addr_Start_out = 0x9000  # 30000  
        prim_soma1.in_row_max = 0
        prim_soma1.mem_sel = 1
        # loop move                        # 2020.07.18修改
        prim_soma1.real_length_in_en = True
        prim_soma1.real_num_in = 1
        prim_in = self.vecter_0()  
        # soma1_entry[0].memory_blocks[0]['data'] = prim_in
        # prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Sending_down_overlap_{}'.format(color),
                'start': prim_soma1.Addr_Start_in,
                # 'length': 448,
                'data':  prim_in,
                'mode': 0,
                'initialize': True},
            # {'name': 'Sending_down_buff',
            #     'start': prim_soma1.Addr_Start_out,
            #     'length': 448,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 55  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 24000
        prim_router.Addr_Dout_length = 27  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base   #  0x300  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400 # 4B寻址 21000
        prim_router.Addr_Din_length = 55   #  83  # 需要减1
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        prim_router.send_destin_core_grp.append({'core_id': ((1,0),(13, 0)), 'data_num': 56, 'T_mode': 1, 'Rhead_num': 1})
        prim_router.recv_source_core_grp.append({'core_id': ((1,0),(1, 0)), 'data_num': 56,'T_mode': 1, 'Rhead_num': 1})

        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_down_overlap_{}'.format(color),
        #         'start': prim_router.Addr_Dout_base,
        #         'data':  prim_in[0],
        #         'mode': 0,
        #         'initialize': False},
        # ]
        prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=0, pack_per_Rhead=55, A_offset=0, Const=55, EN=1)  # 一对一包头参数形式

        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 14*16
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 2
        prim_soma2.num_ciso = 0
        prim_soma2.length_out = 14*16
        prim_soma2.num_out = 2
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8 # 1E6A0
        prim_soma2.Addr_Start_out = s2_start_out   #  0x4428  # 110A0
        prim_soma2.in_row_max = 0
        prim_in = prim_soma1.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Receive_down_overlap_{}'.format(color),
                'start': prim_soma2.Addr_Start_in,
                # 'length': 448,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Receive_down_buff',
            #     'start': prim_soma2.Addr_Start_out,
            #     'length': 448,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        return prim_soma1, prim_router, prim_soma2


    def prim_pool_overlap(self, s1_start_in, s2_start_out, x, y, Addr_Rhead_base=0x300):             # for R,G,B
        # 0x06 move 
        prim_soma1 = Prim_06_move_merge()
        prim_soma1.length_in = 32
        prim_soma1.length_ciso = 0
        prim_soma1.num_in = 64
        prim_soma1.num_ciso = 64
        prim_soma1.length_out = 32
        prim_soma1.num_out = 64
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = s1_start_in      #  1CEA0
        prim_soma1.Addr_Start_ciso = 0x79A8 # 1E6A0
        prim_soma1.Addr_Start_out = 0x9000  # 30000  
        prim_soma1.in_row_max = 0
        prim_soma1.mem_sel = 1
        prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Sending_pool_overlap',
                'start': prim_soma1.Addr_Start_in,
                # 'length': 2048,  # 64*32
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Sending_pool_buff',
            #     'start': prim_soma1.Addr_Start_out,
            #     'length': 2048,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 1
        prim_router.Send_number = 255  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 24000
        prim_router.Addr_Dout_length = 127  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 255  # 需要减1  64*32
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        prim_router.send_destin_core_grp.append({'core_id': ((1,0),(1, 0)), 'data_num': 256,'T_mode': 1, 'Rhead_num': 1})
        prim_router.recv_source_core_grp.append({'core_id': ((1,0),(13, 0)), 'data_num': 256,'T_mode': 1, 'Rhead_num': 1})
            
        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_pool_overlap',
        #         'start': prim_router.Addr_Dout_base,
        #         'data':  prim_in[0],
        #         'mode': 0,
        #         'initialize': False},
        # ]

        prim_router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=0, pack_per_Rhead=255, A_offset=0, Const=255, EN=1)  # 一对一包头参数形式
        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 4*16*32 
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 1
        prim_soma2.num_ciso = 0
        prim_soma2.length_out = 4*16*32
        prim_soma2.num_out = 1
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8 # 1E6A0
        prim_soma2.Addr_Start_out = s2_start_out   #  18EA0
        prim_in = prim_soma1.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Receive_pool_overlap',
                'start': prim_soma2.Addr_Start_in,
                # 'length': 2048,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Receive_pool_buff',
            #     'start': prim_soma2.Addr_Start_out,
            #     'length': 2048,
            #     'data': prim_in[1],
            #     'mode': 0},
        ]

        return prim_soma1, prim_router, prim_soma2

    def auto_cnn81(self, addr=0x7A20):
        Axon_Prim = Prim_41_Axon()
        Axon_Prim.Addr_InA_base = addr   # 224*1 
        Axon_Prim.Addr_InB_base = addr     # weight  
        Axon_Prim.Addr_V_base = addr  
        Axon_Prim.axon_delay = True
        Axon_Prim.L4_num = 7           # 4 ， 9
        Axon_Prim.L5_num = 27
        Axon_Prim.A2S2_mode = 1
        # prim_in = Axon_Prim.init_data()
        # Axon_Prim.memory_blocks = [
        #     {'name': 'Input_conv',
        #         'start': Axon_Prim.Addr_InA_base,
        #         # 'length': 14112,  #  224*21*3
        #         'data':  prim_in[0],
        #         'mode': 0,
        #         'initialize': False},
        #     {'name': 'W_conv',
        #         'start': Axon_Prim.Addr_InB_base,
        #         # 'length': 8064,    #   128*21*3
        #         'data': prim_in[1],
        #         'mode': 0,
        #         'initialize': False},
        # ]
        return Axon_Prim

    def G2_prim_RS2(self, start_out, Addr_Rhead_base, phase=0, N=1):
        # 0x09 rouer        
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 0
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 0
        prim_router.Send_number = 0   # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0   #  4B寻址 
        prim_router.Addr_Dout_length = 0   # 16B的个数，需要减1 
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1 
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000 
        # prim_router.Addr_Din_length = 639   # 需要减1  4*20*64/8 
        prim_router.Receive_number = 39
        if N == 2:
            prim_router.Addr_Din_length = 511   #  4*16*64/8
            prim_router.Receive_number = 31   # 4*8
        else:
            prim_router.Addr_Din_length = 639   # 需要减1  4*20*64/8 
            prim_router.Receive_number = 39   # 4*10
        prim_router.Nx = 0
        prim_router.Ny = 1
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = 0
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = 447
        prim_router.Q = 0
        prim_router.T_mode = 0
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 0

        #  prim_router.send_destin_core_grp.append({'core_id': (1, 1), 'data_num': 84,'T_mode': 1, 'Rhead_num': 1})  4*32
        if N == 1:
            for i in range(5):
                for j in range(2):
                    prim_router.recv_source_core_grp.append({'core_id': ((1,0),(i, j)), 'data_num': 16*4,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
        elif N == 2:
            for i in range(5,9):
                for j in range(2):
                    prim_router.recv_source_core_grp.append({'core_id': ((1,0),(i, j)), 'data_num': 16*4,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
        
        elif N == 3:
            for i in range(9,14):
                for j in range(2):
                    prim_router.recv_source_core_grp.append({'core_id': ((1,0),(i, j)), 'data_num': 16*4,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
                    

        #  move prim 
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 4*64
        prim_soma2.length_out = 4*64
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 20
        prim_soma2.num_out = 20
        prim_soma2.num_ciso = 20
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0 
        prim_soma2.Addr_Start_in = 0x8400     # 21000 
        prim_soma2.Addr_Start_out = start_out     
        prim_soma2.Addr_Start_ciso = 0x79A8   # 1E6A0
        prim_soma2.in_row_max = 0 
        prim2_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Router_receive_g2',
                'start': 0x8400,
                # 'length': 3584,    #  224*16
                'data':  prim2_in[0],
                'mode': 0,
                'initialize': False
                },
        ]

        return prim_router, prim_soma2


    def G2config_core_from_core1(self, core1_cofig, i, j):
        core_cofig = copy.deepcopy(core1_cofig)
        #  增加phase  2020.07.21
        for phase in range(self.group2_phase_num):
            if core_cofig['prims'][phase]['router'] != None:

                if j == 2:
                    pass
                elif j == 3:
                    prim_router, _ = self.G2_prim_RS2(start_out=0x4000, Addr_Rhead_base=0x300, phase=0, N=2)
                    core_cofig['prims'][0]['router'] = [prim_router]
                elif j == 4:
                    prim_router, _ = self.G2_prim_RS2(start_out=0x4000, Addr_Rhead_base=0x300, phase=0, N=3)
                    core_cofig['prims'][0]['router'] = [prim_router]
        
        return core_cofig

    def G2_config_core1(self):
        self.G2_config_prim()

        core1_cofig = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase1-4
               
            },
            'prims': self.prim_list_g2
        }
        return core1_cofig

    def G2_config_prim(self):
        for phase in range(self.group2_phase_num):
            if (phase+1) == 1:            # p1: RBG, p2:GRB, p3: BGR
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x0498)
                # 0x09 rouer + 0x06 merge 
                prim_router, prim_soma2 = self.G2_prim_RS2(start_out=0x0000, Addr_Rhead_base=0x300, phase=phase, N=1)     # 12760
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_g2.append(one_phase_dict)
            
            else:
                self.prim_list_g2.append(self.none_phase())


    def config_prim(self):
        for phase in range(self.group1_phase_num + self.L50_phase_num + self.addd):
            if (phase+1) == 1:            # p1: RBG, p2:GRB, p3: BGR
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x0498)
                # 0x09 rouer +  0x06 merge   clock:2086
                prim_router, prim_soma2 = self.prim_RS2_phase1_4(start_out=0x40A8, Addr_Rhead_base=0x300, idx=0, idy=2, phase=phase)    # 102A0
                # phase dict
                one_phase_dict['axon'] =  [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Router_receive_Phase1_4', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 2:
                one_phase_dict = {}    
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x0498)       
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 3:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x0498)
                # 0x09 rouer + 0x06 merge 
                prim_router, prim_soma2 = self.prim_RS2_phase1_4(start_out=0x49D8, Addr_Rhead_base=0x378, idx=0, idy=4, phase=phase)     # 12760
                # phase dict
                one_phase_dict['axon'] = None #[prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Router_receive_Phase1_4', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 4:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x0498)
                # 0x09 rouer + 0x06 merge
                prim_router, prim_soma2 = self.prim_RS2_phase1_4(start_out=0x4540, Addr_Rhead_base=0x374, idx=0, idy=3, phase=phase)     # 11500
                # phase dict
                one_phase_dict['axon'] = None #[prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Router_receive_Phase1_4', 0)]
                self.prim_list.append(one_phase_dict)           
            
            # R,G,B overlap phase5~10
            elif (phase+1) == 5:
                one_phase_dict = {}
                # top overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_top_overlap(s1_start_in=0x4380, s2_start_out=0x4000, x=1, y=0, color='R', Addr_Rhead_base=0x37C)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_top_overlap_R', 0)]
                one_phase_dict['router'] = [prim_router]     
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_top_overlap_R', 0)]
                self.prim_list.append(one_phase_dict)
            
            elif (phase+1) == 6:
                one_phase_dict = {}
                # down overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_down_overlap(s1_start_in=0x7A18, s2_start_out= 0x4428, x=13, y=0, color='R', Addr_Rhead_base=0x380)
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_down_overlap_R', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_down_overlap_R', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 7:
                one_phase_dict = {}
                # top overlap for G
                prim_soma1, prim_router, prim_soma2 = self.prim_top_overlap(s1_start_in=0x4818, s2_start_out=0x4498, x=1, y=0, color='G', Addr_Rhead_base=0x384)  # 12060  11260
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_top_overlap_G', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_top_overlap_G', 0)]
                self.prim_list.append(one_phase_dict)
            
            elif (phase+1) == 8:
                one_phase_dict = {}
                # down overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_down_overlap(s1_start_in=0x7A18, s2_start_out=0x48C0, x=13, y=0, color='G', Addr_Rhead_base=0x388)  # 1E860  12300
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_down_overlap_G', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_down_overlap_G', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 9:
                one_phase_dict = {}
                # top overlap for B
                prim_soma1, prim_router, prim_soma2 = self.prim_top_overlap(s1_start_in=0x4CB0, s2_start_out=0x4930, x=1, y=0, color='B', Addr_Rhead_base=0x38C)  # 132C0 124C0
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_top_overlap_B', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_top_overlap_B', 0)]
                self.prim_list.append(one_phase_dict)
            
            elif (phase+1) == 10:
                one_phase_dict = {}
                # down overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_down_overlap(s1_start_in=0x7A18, s2_start_out= 0x4D58, x=13, y=0, color='B', Addr_Rhead_base=0x390)  # 1E860  13560
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_down_overlap_B', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_down_overlap_B', 0)]
                self.prim_list.append(one_phase_dict)

            # left calcution
            elif (phase+1) == 11:   # left split
                one_phase_dict = {}
                prim_soma1 = self.prim_split_LR(start_in=0x4000, state='L')
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Split_L', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list.append(one_phase_dict)
            
            # right calcution
            elif (phase+1) == 15:   # right split
                one_phase_dict = {}
                prim_soma1 = self.prim_split_LR(start_in=0x4018, state='R')
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Split_R', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 12:   # left conv
                one_phase_dict = {}
                prim_axon, prim_soma1 = self.prim_conv81_AS1()
                # phase dict
                one_phase_dict['axon'] = [prim_axon, ('Input_conv', 0), ('W_conv', 0)]
                one_phase_dict['soma1'] = [prim_soma1, ('Conv_2row', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 16:   # right conv
                one_phase_dict = {}
                prim_axon, prim_soma1 = self.prim_conv81_AS1()
                # phase dict
                one_phase_dict['axon'] = [prim_axon, ('Input_conv', 0), ('W_conv', 0)]
                one_phase_dict['soma1'] = [prim_soma1, ('Conv_2row', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 13:  #pool overlap
                one_phase_dict = {}
                # top overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_pool_overlap(s1_start_in=0x73A8, s2_start_out=0x63A8, x=1, y=0, Addr_Rhead_base=0x394)  #  1CEA0  18EA0
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_pool_overlap', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_pool_overlap', 0)]
                self.prim_list.append(one_phase_dict)
            
            elif (phase+1) == 17:  #pool overlap
                one_phase_dict = {}
                # top overlap for R
                prim_soma1, prim_router, prim_soma2 = self.prim_pool_overlap(s1_start_in=0x73A8, s2_start_out=0x63A8, x=1, y=0, Addr_Rhead_base=0x398)  #  1CEA0  18EA0
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Sending_pool_overlap', 0)]
                one_phase_dict['router'] = [prim_router]
                one_phase_dict['soma2'] = [prim_soma2, ('Receive_pool_overlap', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 14:  #pool left
                one_phase_dict = {}
                prim_soma1, prim_soma2 = self.prim_pool_LR(s1_start_out=0x8400, s2_start_in=0x8400, s2_start_out=0x76A8, state='L')  # 21000  21000  1D6A0+400（75A8+100）
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Cut', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = [prim_soma2, ('Pool_L', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 18:  #pool right
                one_phase_dict = {}
                prim_soma1, prim_soma2 = self.prim_pool_LR(s1_start_out=0x8B00, s2_start_in=0x8B20, s2_start_out=0x4000, state='R')  # 22C00  22C80  10000
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [prim_soma1, ('Cut', 0)]
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = [prim_soma2, ('Pool_R', 0)]
                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 19:  #merge and send to L2 
                one_phase_dict = {}

                Soma_Prim_06 = Prim_06_move_merge()
                Soma_Prim_06.length_in = 56
                Soma_Prim_06.length_ciso = 56
                Soma_Prim_06.num_in = 4
                Soma_Prim_06.num_ciso = 4
                Soma_Prim_06.length_out = 112
                Soma_Prim_06.num_out = 4
                Soma_Prim_06.type_in = 1
                Soma_Prim_06.type_out = 1
                Soma_Prim_06.in_cut_start = 0
                Soma_Prim_06.Reset_Addr_in = 1
                Soma_Prim_06.Reset_Addr_out = 1
                Soma_Prim_06.Reset_Addr_ciso = 1
                Soma_Prim_06.Row_ck_on = 0
                Soma_Prim_06.Addr_Start_in = 0x75A8   # 1D6A0
                Soma_Prim_06.Addr_Start_ciso = 0x4000 # 10000
                Soma_Prim_06.Addr_Start_out = 0x4400  # 11000
                Soma_Prim_06.in_row_max = 0
                prim_in = Soma_Prim_06.init_data()
                Soma_Prim_06.memory_blocks = [
                    {'name': 'P1_drop',
                        'start': Soma_Prim_06.Addr_Start_in,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': False},
                    {'name': 'P2_drop',
                        'start': Soma_Prim_06.Addr_Start_ciso,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[1],
                        'mode': 0,
                        'initialize': False},
                    # {'name': 'OB_final',
                    #     'start': Soma_Prim_06.Addr_Start_out,
                    #     'length': 7168,     #   56*4*32
                    #     'data':  prim_in[2],
                        # 'mode': 0},
                ]

                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [Soma_Prim_06, ('P1_drop', 0), ('P2_drop', 0)]
                one_phase_dict['router'] =  None           #   [Router_Prim, ('OB_final', 0)]
                one_phase_dict['soma2'] =  None
                self.prim_list.append(one_phase_dict)

            # L50
            elif (phase+1) == 20:  
                one_phase_dict = {}

                Axon_mlp = Prim_04_Axon()
                Axon_mlp.InA_type = 1
                Axon_mlp.InB_type = 1
                Axon_mlp.Load_Bias = 0  # 此处待修改
                Axon_mlp.cin = 1888
                Axon_mlp.cout = 32
                # self.constant_b = 0
                Axon_mlp.Reset_Addr_A = 1
                # Axon_mlp.Reset_Addr_V = 1
                Axon_mlp.Addr_InA_base = 0x7810   # 0x4000   # 10000
                Axon_mlp.Addr_InB_base = 0x0500  # 1400
                Axon_mlp.Addr_Bias_base = 0x7AB0  # 1EAC0
                Axon_mlp.Addr_V_base = 0x4200    # 10800
                Axon_mlp.A2S2_mode = 0

                prim_in = Axon_mlp.init_data()
                Axon_mlp.memory_blocks = [
                    {'name': 'mlp_buff1',
                        'start': Axon_mlp.Addr_InA_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': True},
                    {'name': 'mlp_w1',
                        'start': Axon_mlp.Addr_InB_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[1],
                        'mode': 0,
                        'initialize': True},
                ]

                # phase dict
                one_phase_dict['axon'] = [Axon_mlp, ('mlp_buff1', 0), ('mlp_w1', 0)]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 21:  
                one_phase_dict = {}

                Axon_mlp = Prim_04_Axon()
                Axon_mlp.InA_type = 1
                Axon_mlp.InB_type = 1
                Axon_mlp.Load_Bias = 1  # 此处待修改
                Axon_mlp.cin = 160
                Axon_mlp.cout = 32
                # self.constant_b = 0
                Axon_mlp.Reset_Addr_A = 1
                # Axon_mlp.Reset_Addr_V = 1
                Axon_mlp.Addr_InA_base = 0x79E8          #0x41D8   # 10760
                Axon_mlp.Addr_InB_base = 0x7B00  # 1EC00
                Axon_mlp.Addr_Bias_base = 0x4200
                Axon_mlp.Addr_V_base = 0x4220    # 10880
                Axon_mlp.A2S2_mode = 0

                prim_in = Axon_mlp.init_data()
                Axon_mlp.memory_blocks = [
                    {'name': 'mlp_buff2',
                        'start': Axon_mlp.Addr_InA_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': True},
                    {'name': 'mlp_w2',
                        'start': Axon_mlp.Addr_InB_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[1],
                        'mode': 0,
                        'initialize': True},
                ]

                # relu
                Soma_Prim_05 = Prim_X5_Soma()
                Soma_Prim_05.pad_on = False
                Soma_Prim_05.CMP_C_en = False
                Soma_Prim_05.type_in = 0
                Soma_Prim_05.type_out = 1
                Soma_Prim_05.cin = 32  # 为支持channel数不足16B的情况，所以用户可直接设置cin，程序会自动计算X_Km_num
                Soma_Prim_05.cout = 32
                Soma_Prim_05.Input_fm_Px = 1
                Soma_Prim_05.Input_fm_Py = 1
                Soma_Prim_05.pad_top = 0
                Soma_Prim_05.pad_down = 0
                Soma_Prim_05.pad_left = 0
                Soma_Prim_05.pad_right = 0
                Soma_Prim_05.pooling_Kx = 1
                Soma_Prim_05.pooling_Ky = 1
                Soma_Prim_05.pooling_Sx = 1
                Soma_Prim_05.pooling_Sy = 1
                Soma_Prim_05.CMP_C = 0x0000000 
                Soma_Prim_05.in_cut_start = 0 
                Soma_Prim_05.in_row_max = 2
                Soma_Prim_05.reset_Addr_in = 1
                Soma_Prim_05.reset_Addr_out = 1
                Soma_Prim_05.Addr_Start_in = 0x4200  #  0x10880
                Soma_Prim_05.Addr_Start_out = 0x4240      #0x10900
                Soma_Prim_05.Row_ck_on = 1
                prim_in = Soma_Prim_05.init_data()
                Soma_Prim_05.memory_blocks = [
                    {'name': 'mlp_relu',
                        'start': Soma_Prim_05.Addr_Start_in,
                        # 'length': 16384,    #   64*2*32*4
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': False},
                ]

                one_phase_dict['axon'] = [Axon_mlp, ('mlp_buff1', 0), ('mlp_w1', 0)]
                one_phase_dict['soma1'] = [Soma_Prim_05, ('mlp_relu', 0)]
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list.append(one_phase_dict)

            elif (phase+1) == 22:

                one_phase_dict = {}

                Soma_lut = Prim_07_LUT()
                # Soma_lut.neuron_real_num = 0
                Soma_lut.X_type = 1  # 0:int32 1/2/3 int8
                Soma_lut.Y_type = 1
                Soma_lut.group_num = 1
                Soma_lut.neuron_real_num = 32
                Soma_lut.reset_Addr_X = 1
                Soma_lut.reset_Addr_Y = 1
                Soma_lut.Row_ck_on = 0
                Soma_lut.Addr_X_Start = 0x4240    # 10900
                Soma_lut.Addr_Start_out = 0x8400   #  21000
                Soma_lut.Addr_LUT_Start = 0x7AC0  # 1EB00
                Soma_lut.LUT_DW = 1  # 0:4b   1:8b    2:12b   3:16b
                Soma_lut.X_cut_start = 0
                Soma_lut.in_row_max = 1
                prim_in = Soma_lut.init_data()
                Soma_lut.memory_blocks = [
                    {'name': 'LUT_buff',
                        'start': Soma_lut.Addr_X_Start,
                        # 'length': 16384,    #   64*2*32*4
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': False},
                ]

                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [Soma_lut, ('LUT_buff', 0)]
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list.append(one_phase_dict)

            else:
                self.prim_list.append(self.none_phase())

    def config_prim_extra(self):
        for phase in range(self.group1_phase_num + self.L50_phase_num + self.addd):
            # L50
            if (phase+1) == 20:  
                one_phase_dict = {}

                Axon_mlp = Prim_04_Axon()
                Axon_mlp.InA_type = 1
                Axon_mlp.InB_type = 1
                Axon_mlp.Load_Bias = 0  # 此处待修改
                Axon_mlp.cin = 1888
                Axon_mlp.cout = 32
                # self.constant_b = 0
                Axon_mlp.Reset_Addr_A = 1
                # Axon_mlp.Reset_Addr_V = 1
                Axon_mlp.Addr_InA_base = 0x4000   # 10000
                Axon_mlp.Addr_InB_base = 0x0500  # 1400
                Axon_mlp.Addr_Bias_base = 0x7AB0  # 1EAC0
                Axon_mlp.Addr_V_base = 0x4200    # 10800
                Axon_mlp.A2S2_mode = 0

                prim_in = Axon_mlp.init_data()
                Axon_mlp.memory_blocks = [
                    {'name': 'mlp_buff1',
                        'start': Axon_mlp.Addr_InA_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': True},
                    {'name': 'mlp_w1',
                        'start': Axon_mlp.Addr_InB_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[1],
                        'mode': 0,
                        'initialize': True},
                ]

                # phase dict
                one_phase_dict['axon'] = [Axon_mlp, ('mlp_buff1', 0), ('mlp_w1', 0)]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list_extra.append(one_phase_dict)

            elif (phase+1) == 21:  
                one_phase_dict = {}

                Axon_mlp = Prim_04_Axon()
                Axon_mlp.InA_type = 1
                Axon_mlp.InB_type = 1
                Axon_mlp.Load_Bias = 1  # 此处待修改
                Axon_mlp.cin = 160
                Axon_mlp.cout = 32
                # self.constant_b = 0
                Axon_mlp.Reset_Addr_A = 1
                # Axon_mlp.Reset_Addr_V = 1
                Axon_mlp.Addr_InA_base = 0x41D8   # 10760
                Axon_mlp.Addr_InB_base = 0x7B00  # 1EC00
                Axon_mlp.Addr_Bias_base = 0x4200 # 1EAC0
                Axon_mlp.Addr_V_base = 0x4220    # 10880
                Axon_mlp.A2S2_mode = 0

                prim_in = Axon_mlp.init_data()
                Axon_mlp.memory_blocks = [
                    {'name': 'mlp_buff2',
                        'start': Axon_mlp.Addr_InA_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': True},
                    {'name': 'mlp_w2',
                        'start': Axon_mlp.Addr_InB_base,
                        # 'length': 3584,     #   28*4*32
                        'data':  prim_in[1],
                        'mode': 0,
                        'initialize': True},
                ]

                # relu
                Soma_Prim_05 = Prim_X5_Soma()
                Soma_Prim_05.pad_on = False
                Soma_Prim_05.CMP_C_en = False
                Soma_Prim_05.type_in = 0
                Soma_Prim_05.type_out = 1
                Soma_Prim_05.cin = 32  # 为支持channel数不足16B的情况，所以用户可直接设置cin，程序会自动计算X_Km_num
                Soma_Prim_05.cout = 32
                Soma_Prim_05.Input_fm_Px = 1
                Soma_Prim_05.Input_fm_Py = 1
                Soma_Prim_05.pad_top = 0
                Soma_Prim_05.pad_down = 0
                Soma_Prim_05.pad_left = 0
                Soma_Prim_05.pad_right = 0
                Soma_Prim_05.pooling_Kx = 1
                Soma_Prim_05.pooling_Ky = 1
                Soma_Prim_05.pooling_Sx = 1
                Soma_Prim_05.pooling_Sy = 1
                Soma_Prim_05.CMP_C = 0x0000000 
                Soma_Prim_05.in_cut_start = 0 
                Soma_Prim_05.in_row_max = 2
                Soma_Prim_05.reset_Addr_in = 1
                Soma_Prim_05.reset_Addr_out = 1
                Soma_Prim_05.Addr_Start_in = 0x4200  #  0x10880
                Soma_Prim_05.Addr_Start_out = 0x4240      #0x10900
                Soma_Prim_05.Row_ck_on = 1
                prim_in = Soma_Prim_05.init_data()
                Soma_Prim_05.memory_blocks = [
                    {'name': 'mlp_relu',
                        'start': Soma_Prim_05.Addr_Start_in,
                        # 'length': 16384,    #   64*2*32*4
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': False},
                ]

                one_phase_dict['axon'] = [Axon_mlp, ('mlp_buff1', 0), ('mlp_w1', 0)]
                one_phase_dict['soma1'] = [Soma_Prim_05, ('mlp_relu', 0)]
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list_extra.append(one_phase_dict)

            elif (phase+1) == 22:

                one_phase_dict = {}

                Soma_lut = Prim_07_LUT()
                # Soma_lut.neuron_real_num = 0
                Soma_lut.X_type = 1  # 0:int32 1/2/3 int8
                Soma_lut.Y_type = 1
                Soma_lut.group_num = 1
                Soma_lut.neuron_real_num = 32
                Soma_lut.reset_Addr_X = 1
                Soma_lut.reset_Addr_Y = 1
                Soma_lut.Row_ck_on = 0
                Soma_lut.Addr_X_Start = 0x4240    # 10900
                Soma_lut.Addr_Start_out = 0x8400   #  21000
                Soma_lut.Addr_LUT_Start = 0x7AC0  # 1EB00
                Soma_lut.LUT_DW = 1  # 0:4b   1:8b    2:12b   3:16b
                Soma_lut.X_cut_start = 0
                Soma_lut.in_row_max = 1
                prim_in = Soma_lut.init_data()
                Soma_lut.memory_blocks = [
                    {'name': 'LUT_buff',
                        'start': Soma_lut.Addr_X_Start,
                        # 'length': 16384,    #   64*2*32*4
                        'data':  prim_in[0],
                        'mode': 0,
                        'initialize': False},
                ]


                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = [Soma_lut, ('LUT_buff', 0)]
                one_phase_dict['router'] =  None
                one_phase_dict['soma2'] =  None

                self.prim_list_extra.append(one_phase_dict)  

            else:     
                one_phase_dict = {}
                # phase dict
                one_phase_dict['axon'] = None
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = None
                one_phase_dict['soma2'] = None
                self.prim_list_extra.append(one_phase_dict)   

                
    def prim_pool_LR(self, s1_start_out, s2_start_in, s2_start_out, state='L'):

        Soma_Prim_05 = Prim_X5_Soma()
        Soma_Prim_05.pad_on = True
        Soma_Prim_05.CMP_C_en = True
        Soma_Prim_05.type_in = 1
        Soma_Prim_05.type_out = 1
        Soma_Prim_05.cin = 32  # 为支持channel数不足16B的情况，所以用户可直接设置cin，程序会自动计算X_Km_num
        Soma_Prim_05.cout = 32
        Soma_Prim_05.Input_fm_Px = 64
        Soma_Prim_05.Input_fm_Py = 8
        Soma_Prim_05.pad_top = 0
        Soma_Prim_05.pad_down = 0
        Soma_Prim_05.pad_left = 1
        Soma_Prim_05.pad_right = 0
        Soma_Prim_05.pooling_Kx = 3
        Soma_Prim_05.pooling_Ky = 3
        Soma_Prim_05.pooling_Sx = 2
        Soma_Prim_05.pooling_Sy = 2
        Soma_Prim_05.CMP_C = 0x0000000
        Soma_Prim_05.in_cut_start = 0
        Soma_Prim_05.in_row_max = 0
        Soma_Prim_05.reset_Addr_in = 1  # unsure
        Soma_Prim_05.reset_Addr_out = 1  # UNSURE
        Soma_Prim_05.Addr_Start_in = 0x63A8     #0x18EA0
        Soma_Prim_05.Addr_Start_out = s1_start_out      #0x21000 22C00
        Soma_Prim_05.Row_ck_on = 0
        prim_in = Soma_Prim_05.init_data()
        Soma_Prim_05.memory_blocks = [
            {'name': 'Cut',
                'start': Soma_Prim_05.Addr_Start_in,
                # 'length': 16384,    #   64*8*32
                'data': prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Pool1',
            #     'start': Soma_Prim_05.Addr_Start_out,
            #     'length': 4096,     #   32*4*32
            #     'data':  prim_in[1],
                # 'mode': 0},
        ]


        Soma_Prim = Prim_06_move_split()
        Soma_Prim.length_in = 32*32
        Soma_Prim.num_in = 4
        Soma_Prim.length_out = 28*32
        Soma_Prim.length_ciso = 0
        Soma_Prim.num_out = 4
        Soma_Prim.num_ciso = 4
        Soma_Prim.type_in = 1
        Soma_Prim.type_out = 1
        Soma_Prim.in_cut_start = 0
        Soma_Prim.Reset_Addr_in = 1
        Soma_Prim.Reset_Addr_out = 1
        Soma_Prim.Reset_Addr_ciso = 1
        Soma_Prim.Row_ck_on = 0
        Soma_Prim.Addr_Start_in = s2_start_in  #21000    22C80
        Soma_Prim.Addr_Start_ciso = 0x7A90  #1EA40
        Soma_Prim.Addr_Start_out = s2_start_out   #1D6A0  10000
        Soma_Prim.in_row_max = 0
        prim_in = Soma_Prim.init_data()
        Soma_Prim.memory_blocks = [
            {'name': 'Pool_{}'.format(state),
                'start': Soma_Prim.Addr_Start_in,
                # 'length': 4096,     #   32*4*32
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'P1_drop',
            #     'start': Soma_Prim.Addr_Start_out,
            #     'length': 3584,     #   28*4*32
            #     'data':  prim_in[1],
                # 'mode': 0},
        ]

        return Soma_Prim_05, Soma_Prim


    def prim_conv81_AS1(self):
        #conv81
        Axon_Prim = Prim_81_Axon()
        Axon_Prim.pad_on = True
        Axon_Prim.InA_type = 1      #C++仿真器不支持InA_type = 3
        Axon_Prim.InB_type = 1
        Axon_Prim.Load_Bias = 0
        Axon_Prim.cin = 3
        Axon_Prim.cout = 32
        Axon_Prim.Input_fm_Px = 128
        Axon_Prim.Input_fm_Py = 21
        Axon_Prim.pad_up = 0
        Axon_Prim.pad_down = 0
        Axon_Prim.pad_left = 3
        Axon_Prim.pad_right = 2
        Axon_Prim.conv_Kx = 7
        Axon_Prim.conv_Ky = 7
        Axon_Prim.conv_Sx = 2
        Axon_Prim.conv_Sy = 2
        Axon_Prim.conv_Ex = 1
        Axon_Prim.conv_Ey = 1
        Axon_Prim.Reset_Addr_A = 1
        Axon_Prim.Reset_Addr_V = 1
        Axon_Prim.Addr_InA_base = 0x4DE0  #  0x4DC8 + 18              #0x13720
        Axon_Prim.Addr_InB_base = 0x0000     # weight split  //not sure
        Axon_Prim.Addr_Bias_base = 0x79A8     #0x1E6A0
        Axon_Prim.Addr_V_base = 0x57C0  #  0x57A8+18      #0x156A0补上padding内存  55A8+200
        prim_in = Axon_Prim.init_data()
        Axon_Prim.memory_blocks = [
            {'name': 'Input_conv',
                'start': Axon_Prim.Addr_InA_base,
                # 'length': 14112,  #  224*21*3
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},    
            {'name': 'W_conv',
                'start': Axon_Prim.Addr_InB_base,
                # 'length': 8064,    #   128*21*3
                'data': prim_in[1],
                'mode': 0,
                'initialize': True},
            # {'name': 'Conv_2row',
            #     'start':  Axon_Prim.Addr_V_base,
            #     'length': 16384,    #   64*2*32*4
            #     'data': prim_in[2],
                # 'mode': 0},
        ]

        # relu
        Soma_Prim_05 = Prim_X5_Soma()
        Soma_Prim_05.pad_on = False
        Soma_Prim_05.CMP_C_en = False
        Soma_Prim_05.type_in = 0
        Soma_Prim_05.type_out = 1
        Soma_Prim_05.cin = 32  # 为支持channel数不足16B的情况，所以用户可直接设置cin，程序会自动计算X_Km_num
        Soma_Prim_05.cout = 32
        Soma_Prim_05.Input_fm_Px = 64
        Soma_Prim_05.Input_fm_Py = 8
        Soma_Prim_05.pad_top = 0
        Soma_Prim_05.pad_down = 0
        Soma_Prim_05.pad_left = 0
        Soma_Prim_05.pad_right = 0
        Soma_Prim_05.pooling_Kx = 1
        Soma_Prim_05.pooling_Ky = 1
        Soma_Prim_05.pooling_Sx = 1
        Soma_Prim_05.pooling_Sy = 1
        Soma_Prim_05.CMP_C = 0x0000000 
        Soma_Prim_05.in_cut_start = 7
        Soma_Prim_05.in_row_max = 1 
        Soma_Prim_05.reset_Addr_in = 1
        Soma_Prim_05.reset_Addr_out = 1
        Soma_Prim_05.Addr_Start_in = 0x57C0  #  0x57A8+18      #0x156A0 后移补padding的地址
        Soma_Prim_05.Addr_Start_out = 0x65A8      #0x196A0
        Soma_Prim_05.Row_ck_on = 1
        prim_in = Soma_Prim_05.init_data()
        Soma_Prim_05.memory_blocks = [
            {'name': 'Conv_2row',
                'start': Soma_Prim_05.Addr_Start_in,
                # 'length': 16384,    #   64*2*32*4
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Cut',
            #     'start': Soma_Prim_05.Addr_Start_out,
            #     'length': 16384,    #   64*8*32
            #     'data': prim_in[1],
                # 'mode': 0},
        ]

        return Axon_Prim, Soma_Prim_05


    def prim_split_LR(self, start_in, state='L'):
        prim_soma1 = Prim_06_move_split()
        prim_soma1.length_in = 14*16
        prim_soma1.num_in = 63
        prim_soma1.length_out = 8*16
        prim_soma1.length_ciso = 6*16
        prim_soma1.num_out = 63
        prim_soma1.num_ciso = 63
        prim_soma1.type_in = 1
        prim_soma1.type_out = 1
        prim_soma1.in_cut_start = 0
        prim_soma1.Reset_Addr_in = 1
        prim_soma1.Reset_Addr_out = 1
        prim_soma1.Reset_Addr_ciso = 1
        prim_soma1.Row_ck_on = 0
        prim_soma1.Addr_Start_in = start_in  #10000   R 10006
        prim_soma1.Addr_Start_ciso = 0x8400  #21000   R 21000
        prim_soma1.Addr_Start_out = 0x4DE0   # 0x4DC8+18   #13720   R 13720
        prim_soma1.in_row_max = 0
        prim_in = prim_soma1.init_data()
        prim_soma1.memory_blocks = [
            {'name': 'Split_{}'.format(state),
                'start': prim_soma1.Addr_Start_in,
                # 'length': 14112,  #  224*21*3
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
            # {'name': 'Split_end',
            #     'start': prim_soma1.Addr_Start_out,
            #     'length': 8064,    #   128*21*3
            #     'data': prim_in[1],
                # 'mode': 0},
        ]

        return prim_soma1
           

    def config_core1(self):
        self.config_prim()

        core1_cofig = {  # core id     # length先不计算
            'memory_blocks': {
                #  phase1-4
                'Router_receive_Phase1_4': {'start': 0x8400, 'length': 1, 'initialize': False},
                #  phase5-10
                'Sending_top_overlap_R': {'start': 0x4380, 'length': 1, 'initialize': False},
                # 'Router_top_overlap_R': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_top_overlap_R': {'start': 0x8400, 'length': 1, 'initialize': False},
                'Sending_down_overlap_R': {'start': 0x7A18, 'length': 1, 'initialize': False},
                # 'Router_down_overlap_R': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_down_overlap_R': {'start': 0x8400, 'length': 1, 'initialize': False},

                'Sending_top_overlap_G': {'start': 0x4818, 'length': 1, 'initialize': False},
                # 'Router_top_overlap_G': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_top_overlap_G': {'start': 0x8400, 'length': 1, 'initialize': False},
                'Sending_down_overlap_G': {'start': 0x7A18,'length': 1, 'initialize': False},
                # 'Router_down_overlap_G': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_down_overlap_G': {'start': 0x8400, 'length': 1, 'initialize': False},

                'Sending_top_overlap_B': {'start': 0x4CB0, 'length': 1, 'initialize': False},
                # 'Router_top_overlap_B': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_top_overlap_B': {'start': 0x8400, 'length': 1, 'initialize': False},
                'Sending_down_overlap_B': {'start': 0x7A18, 'length': 1, 'initialize': False},
                # 'Router_down_overlap_B': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_down_overlap_B': {'start': 0x8400, 'length': 1, 'initialize': False},
                # Split
                'Split_L': {'start': 0x4000, 'length': 1, 'initialize': False},
                'Split_R': {'start': 0x4001, 'length': 1, 'initialize': False},
                # Conv
                'Input_conv' : {'start': 0x4DC8, 'length': 1, 'initialize': False},
                'W_conv' : {'start': 0x0000, 'length': 1, 'initialize': True},
                'Conv_2row': {'start':0x55A8, 'length': 1, 'initialize': False},
                #  pooling overlap
                'Sending_pool_overlap': {'start': 0x73A8, 'length': 1, 'initialize': False},
                # 'Router_pool_overlap': {'start': 0x9000, 'length': 1, 'initialize': False},
                'Receive_pool_overlap': {'start': 0x8400, 'length': 1, 'initialize': False},
                # Pool and drop
                'Cut': {'start':0x63A8, 'length': 1, 'initialize': False},
                'Pool_L' : {'start':0x8400, 'length': 1, 'initialize': False},
                'Pool_R' : {'start':0x8B20, 'length': 1, 'initialize': False},
                'P1_drop': {'start': 0x75A8, 'length': 1, 'initialize': False},
                'P2_drop': {'start': 0x4000, 'length': 1, 'initialize': False},
                'OB_final' : {'start': 0x4400, 'length': 1, 'initialize': False},
                # L50
                'mlp_buff1': {'start':0x4000, 'length': 1, 'initialize': False},
                'mlp_w1' : {'start':0x0500, 'length': 1, 'initialize': False},
                'mlp_buff2' : {'start': 0x41D8 , 'length': 1, 'initialize': False},
                'mlp_w2': {'start': 0x7B00, 'length': 1, 'initialize': False},
                'mlp_relu': {'start': 0x4200, 'length': 1, 'initialize': False},
                'LUT_buff': {'start': 0x4240, 'length': 1, 'initialize': False},
            },
            'prims': self.prim_list
        }
        return core1_cofig

    def config_core1_extra(self):
        self.config_prim_extra()

        core1_cofig_extra = {  # core id     # length先不计算
            'memory_blocks': {
                # L50
                'mlp_buff1': {'start':0x4000, 'length': 1, 'initialize': False},
                'mlp_w1' : {'start':0x0500, 'length': 1, 'initialize': False},
                'mlp_buff2' : {'start': 0x41D8 , 'length': 1, 'initialize': False},
                'mlp_w2': {'start': 0x7B00, 'length': 1, 'initialize': False},
                'mlp_relu': {'start': 0x4200, 'length': 1, 'initialize': False},
                'LUT_buff': {'start': 0x4240, 'length': 1, 'initialize': False},
            },
            'prims': self.prim_list_extra
        }
        return core1_cofig_extra

    def config_core_from_core1(self, core1_cofig, i, j):
        core_cofig = copy.deepcopy(core1_cofig)
        #  code test
        # for phase in range(3):
        #     if i == 0 and j == 2:
        #         a = core_cofig['prims'][phase]['router']
        #         a[0].Ny = 0
        #         print('****{}*****'.format(phase), core1_cofig['prims'][phase]['router'][0].Ny)
        #         print('****{}*****'.format(phase), core_cofig['prims'][phase]['router'][0].Ny)

        for phase in range(4):     # transform phase1~3             
            # if i in range(1,4):    # core(1:3,n)
            #     pass
            if i in range(4,8):   # core(4:7,n)
                if phase == 0:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][3])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 1:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][0])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 2:           # 空phase
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][1])
                    core_cofig['prims'][phase] = phase_core1
                    # if j == 2:
                    #     router_entry = core_cofig['prims'][phase]['router']
                    #     router_entry[0].Ny = 0
                    #     router_entry[0].CXY = 0b00
                elif phase == 3:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][2])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00

            elif i in range(8,12):      # core(8:11,n)
                if phase == 0:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][2])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 1:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][3])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 2:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][0])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 3:           # 空phase
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][1])
                    core_cofig['prims'][phase] = phase_core1

            elif i in range(12,14):      # core(12:13,n)
                if phase == 0:    # 空phase
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][1])
                    core_cofig['prims'][phase] = phase_core1
                elif phase == 1:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][2])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 2:
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][3])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                elif phase == 3:          
                    phase_core1 = copy.deepcopy(core1_cofig['prims'][0])
                    core_cofig['prims'][phase] = phase_core1
                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00

            elif i in range(0,4):
                if j == 1:
                    if phase == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                    else:
                        # print('****(0,1)-phase{}*****'.format(phase), core1_cofig['prims'][phase])
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].Ny = 0
                        router_entry[0].CXY = 0b00
                        # print('****{}*****'.format(phase), core_cofig['prims'][phase]['router'][0].Ny)
                    

        for phase in range(4,10):     # transform phase4~9
            if phase == 4:
                if i == 13:         
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x7A18  #1E860
                    # prim_in = prim_soma1.init_data()
                    prim_in = self.vecter_0()

                    soma1_entry[0].memory_blocks[0]['start'] = 0x7A18 
                    soma1_entry[0].memory_blocks[0]['data'] = prim_in
                    soma1_entry[0].memory_blocks[0]['initialize'] = True

                    # loop move
                    soma1_entry[0].real_length_in_en = True
                    soma1_entry[0].real_num_in = 1
                    # router
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-13, Y=0, A=0, pack_per_Rhead=83, A_offset=0, Const=83, EN=1)

            elif phase == 5:
                if i in range(1,14):
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x40A8  #102A0

                    soma1_entry[0].memory_blocks[0]['start'] = soma1_entry[0].Addr_Start_in 
                    # soma1_entry[0].memory_blocks[0]['data'] = prim_in     # 这里不初始化数据
                    soma1_entry[0].memory_blocks[0]['initialize'] = False
                    # loop move
                    soma1_entry[0].real_length_in_en = False
                    # router
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=55, A_offset=0, Const=55, EN=1)
                # elif i == 0:
                #     soma1_entry = core_cofig['prims'][phase]['soma1']
                #     soma1_entry.Addr_start_in = 0x7A18  #1E860
                #     router_entry = core_cofig['prims'][phase]['router']
                #     router_entry.prim_router.addRHead(S=0, T=1, P=0, Q=0, X=13, Y=0, A=0, pack_per_Rhead=56, A_offset=0, Const=56, EN=1)
            elif phase == 6:
                if i == 13:         
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x7A18  #1E860
                    prim_in = self.vecter_0()

                    soma1_entry[0].memory_blocks[0]['start'] = 0x7A18 
                    soma1_entry[0].memory_blocks[0]['data'] = prim_in
                    soma1_entry[0].memory_blocks[0]['initialize'] = True 

                    # loop move
                    soma1_entry[0].real_length_in_en = True
                    soma1_entry[0].real_num_in = 1
                    # router
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-13, Y=0, A=0, pack_per_Rhead=83, A_offset=0, Const=83, EN=1)
            elif phase == 7:
                if i in range(1,14):
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x4540  #11500

                    soma1_entry[0].memory_blocks[0]['start'] = soma1_entry[0].Addr_Start_in 
                    # soma1_entry[0].memory_blocks[0]['data'] = prim_in     # 这里不初始化数据
                    soma1_entry[0].memory_blocks[0]['initialize'] = False
                    # loop move
                    soma1_entry[0].real_length_in_en = False
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=55, A_offset=0, Const=55, EN=1)
                # elif i == 0:
                #     soma1_entry = core_cofig['prims'][phase]['soma1']
                #     soma1_entry.Addr_start_in = 0x7A18  #1E860
                #     router_entry = core_cofig['prims'][phase]['router']
                #     router_entry.prim_router.addRHead(S=0, T=1, P=0, Q=0, X=13, Y=0, A=0, pack_per_Rhead=56, A_offset=0, Const=56, EN=1)
            elif phase == 8:
                if i == 13:         
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x7A18  #1E860
                    prim_in = self.vecter_0()

                    soma1_entry[0].memory_blocks[0]['start'] = 0x7A18 
                    soma1_entry[0].memory_blocks[0]['data'] = prim_in
                    soma1_entry[0].memory_blocks[0]['initialize'] = True

                    # loop move
                    soma1_entry[0].real_length_in_en = True
                    soma1_entry[0].real_num_in = 1
                    # router
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-13, Y=0, A=0, pack_per_Rhead=83, A_offset=0, Const=83, EN=1)
            elif phase == 9:
                if i in range(1,14):
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x49D8  #12760

                    soma1_entry[0].memory_blocks[0]['start'] = soma1_entry[0].Addr_Start_in 
                    # soma1_entry[0].memory_blocks[0]['data'] = prim_in     # 这里不初始化数据
                    soma1_entry[0].memory_blocks[0]['initialize'] = False
                    # loop move
                    soma1_entry[0].real_length_in_en = False
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=55, A_offset=0, Const=55, EN=1)
                # elif i == 0:
                #     soma1_entry = core_cofig['prims'][phase]['soma1']
                #     soma1_entry.Addr_start_in = 0x7A18  #1E860
                #     router_entry = core_cofig['prims'][phase]['router']
                #     router_entry.prim_router.addRHead(S=0, T=1, P=0, Q=0, X=13, Y=0, A=0, pack_per_Rhead=56, A_offset=0, Const=56, EN=1)

        for phase in range(10,19):     # transform phase11~18
            if phase == 12:
                if i == 13:
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x7A18  #1E860
                    prim_in = self.vecter_01()

                    soma1_entry[0].memory_blocks[0]['start'] = 0x7A18 
                    soma1_entry[0].memory_blocks[0]['data'] = prim_in
                    soma1_entry[0].memory_blocks[0]['initialize'] = True

                    # loop move
                    soma1_entry[0].real_length_in_en = True
                    soma1_entry[0].real_num_in = 1
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-13, Y=0, A=0, pack_per_Rhead=255, A_offset=0, Const=255, EN=1)
            elif phase == 16:
                if i == 13:
                    soma1_entry = core_cofig['prims'][phase]['soma1']
                    soma1_entry[0].Addr_Start_in = 0x79A8  #  0x7A18  #1E860
                    prim_0 = self.vecter_01()

                    soma1_entry[0].memory_blocks[0]['start'] = 0x79A8  #  0x7A18 
                    soma1_entry[0].memory_blocks[0]['data'] = prim_0
                    soma1_entry[0].memory_blocks[0]['initialize'] = True

                    # loop move
                    soma1_entry[0].real_length_in_en = True
                    soma1_entry[0].real_num_in = 2

                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].RHeadList = []
                    router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=-13, Y=0, A=0, pack_per_Rhead=255, A_offset=0, Const=255, EN=1)
        
        # for phase in range(19,21):     # transform phase11~18
        #     if j == 2:
        #         if phase == 19:
        #             phase_core1 = copy.deepcopy(core1_cofig['prims'][20])
        #             core_cofig['prims'][phase] = phase_core1 
        #         elif phase == 20:
        #             phase_core1 = copy.deepcopy(core1_cofig['prims'][19])
        #             core_cofig['prims'][phase] = phase_core1
        #             router_entry = core_cofig['prims'][phase]['router']
        #             # router_entry[0].Ny = -2
        #             router_entry[0].RHeadList = []
        #             router_entry[0].addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=0, pack_per_Rhead=895, A_offset=0, Const=895, EN=1)

        
        for phase in range(4):      # p1: RBG, p2:GRB, p3: BGR
            if phase == 0:
                if i == 0:

                    if j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,2)) #R-C1
                        router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(1,4):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,2))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(4,8):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,3))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(8,12):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,4))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                    
            elif phase == 1:
                if i == 0:

                    if j == 1:
                        pass
                elif i in range(1,4):
                    pass
                elif i in range(4,8):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,2))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(8,12):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,3))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(12,14):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,4))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
            
            elif phase == 2:
                if i in range(0,4):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,4))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(4,8):
                    pass
                elif i in range(8,12):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,2))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(12,14):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,3))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
            
            elif phase == 3:
                if i in range(0,4):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,3))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(4,8):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,4))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase
                elif i in range(8,12):
                    pass
                elif i in range(12,14):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,2))
                    router_entry[0].recv_source_core_grp[0]['sync_phase_num'] = phase

        
        for phase in range(4, 19):
            if (phase == 4) or (phase == 6) or (phase == 8) or (phase == 12) or (phase == 16):
                if i == 0:
                    if j == 0:
                        pass
                    elif j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(1,1))
                        router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(13,1))
                elif i in range(1,13):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(i+1,j))
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(i-1,j))
                elif i == 13:
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(0,j))
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(12,j))
                    
            elif phase == 5 or phase == 7 or phase == 9:
                if i == 0:
                    if j == 0:
                        pass
                    elif j == 1:
                        router_entry = core_cofig['prims'][phase]['router']
                        router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(13,1))
                        router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(1,1))
                elif i in range(1,13):
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(i-1,j))
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(i+1,j))
                elif i == 13:
                    router_entry = core_cofig['prims'][phase]['router']
                    router_entry[0].send_destin_core_grp[0]['core_id'] = ((1,0),(12,j))
                    router_entry[0].recv_source_core_grp[0]['core_id'] = ((1,0),(0,j))
                    
        # for phase in range(19,21):     
        #     if phase == 18:
        #         if j == 1:
        #             if i == 1:
        #                 router_entry = core_cofig['prims'][phase]['router']  
        #                 router_entry[0].send_destin_core_grp[0]['core_id'] = (1,0)
        #             elif i == 2:
        #                 router_entry = core_cofig['prims'][phase]['router'] 
        #                 router_entry[0].send_destin_core_grp[0]['core_id'] = (2,0) 

        #     elif phase == 19:
        #         if j == 2:
        #             if i == 1:
        #                 router_entry = core_cofig['prims'][phase]['router']
        #                 router_entry[0].send_destin_core_grp[0]['core_id'] = (1,0)
        #             elif i == 2:
        #                 router_entry = core_cofig['prims'][phase]['router'] 
        #                 router_entry[0].send_destin_core_grp[0]['core_id'] = (2,0) 

        for phase in range(1):
            if phase == 0:
                if i <= 4 and j == 0:
                    if i < 4:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x40A8, Addr_Rhead_base=0x300, idx=0, idy=2, idxs=2, idys=2, x=2-i, y=2-j, phase=phase, Ai=i*4)
                    elif i == 4:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x4540, Addr_Rhead_base=0x300, idx=0, idy=2, idxs=2, idys=2, x=2-i, y=2-j, phase=phase, Ai=i*4)
                    core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                    core_cofig['prims'][phase]['router'] = [prim_router]
                    core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                elif i<= 4 and j == 1:
                    if i < 4:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x40A8, Addr_Rhead_base=0x300, idx=0, idy=2, idxs=2, idys=2, x=2-i, y=2-j, phase=phase, Ai=i*4+32/(64*4))
                    elif i == 4:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x4540, Addr_Rhead_base=0x300, idx=0, idy=2, idxs=2, idys=2, x=2-i, y=2-j, phase=phase, Ai=i*4+32/(64*4))                                           
                    core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                    core_cofig['prims'][phase]['router'] = [prim_router]
                    core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                    core_cofig['prims'][phase]['router'][0].Ny = 0
                    core_cofig['prims'][phase]['router'][0].CXY = 0b00

                elif i>=9 and j == 0:
                    if i < 12:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x49D8, Addr_Rhead_base=0x300, idx=0, idy=4, idxs=2, idys=4, x=2-i, y=4-j, phase=phase, Ai=i*4)
                        core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                        core_cofig['prims'][phase]['router'] = [prim_router]
                        core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                    elif i >= 12:  #  只发不收
                        prim_soma1, prim_router = self.prim_L1L2_arrange_s(start_in=0x4400, Addr_Rhead_base=0x300, idx=0, idy=4, idxs=2, idys=4, x=2-i, y=4-j, phase=phase, Ai=i*4)
                        core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                        core_cofig['prims'][phase]['router'] = [prim_router]
                elif i>=9 and j == 1:
                    if i < 12:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x49D8, Addr_Rhead_base=0x300, idx=0, idy=4, idxs=2, idys=4, x=2-i, y=4-j, phase=phase, Ai=i*4+32/(64*4))
                        core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                        core_cofig['prims'][phase]['router'] = [prim_router]
                        core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                        core_cofig['prims'][phase]['router'][0].Ny = 0
                        core_cofig['prims'][phase]['router'][0].CXY = 0b00
                    elif i >= 12:
                        prim_soma1, prim_router = self.prim_L1L2_arrange_s(start_in=0x4400, Addr_Rhead_base=0x300, idx=0, idy=4, idxs=2, idys=4, x=2-i, y=4-j, phase=phase, Ai=i*4+32/(64*4))
                        core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                        core_cofig['prims'][phase]['router'] = [prim_router]


                elif i>=5 and i<9 and j == 0:
                    if i < 8:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x4540, Addr_Rhead_base=0x300, idx=0, idy=3, idxs=2, idys=3, x=2-i, y=3-j, phase=phase, Ai=i*4)
                    elif i >= 8:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x49D8, Addr_Rhead_base=0x300, idx=0, idy=3, idxs=2, idys=3, x=2-i, y=3-j, phase=phase, Ai=i*4)
                    core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                    core_cofig['prims'][phase]['router'] = [prim_router]
                    core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                elif i>=5 and i<9 and j == 1:
                    if i < 8:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x4540, Addr_Rhead_base=0x300, idx=0, idy=3, idxs=2, idys=3, x=2-i, y=3-j, phase=phase, Ai=i*4+32/(64*4))
                    elif i >= 8:
                        prim_soma1, prim_router, prim_soma2 = self.prim_L1L2_arrange(start_in=0x4400, start_out=0x49D8, Addr_Rhead_base=0x300, idx=0, idy=3, idxs=2, idys=3, x=2-i, y=3-j, phase=phase, Ai=i*4+32/(64*4))
                    core_cofig['prims'][phase]['soma1'] = [prim_soma1]
                    core_cofig['prims'][phase]['router'] = [prim_router]
                    core_cofig['prims'][phase]['soma2'] = [prim_soma2]
                    core_cofig['prims'][phase]['router'][0].Ny = 0
                    core_cofig['prims'][phase]['router'][0].CXY = 0b00
                                                
        return core_cofig

    def vecter_0(self):
        prim_in = []
        for aaa in range(224 // 4):  #  // 4
            tmp=[]  
            for bbb in range(4):
                tmp.append(0)
            prim_in.append(tmp)
        return prim_in

    def vecter_01(self):
        prim_in = []
        for aaa in range(64 // 4):   #  // 4
            tmp=[]  
            for bbb in range(4):
                tmp.append(0)
            prim_in.append(tmp)
        return prim_in

    
    def prim_source_receive(self, start_out, idx ,idy):
        # 0x09 rouer        
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 0
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 0
        prim_router.Send_number = 0  # 发送的所有包数，包含EN=0的包，需要减1
        prim_router.Addr_Dout_base = 0  #  4B寻址 
        prim_router.Addr_Dout_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = 0  # 4B寻址
        prim_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 895   # 需要减1  56*4*32
        prim_router.Receive_number = 0
        prim_router.Nx = 0
        prim_router.Ny = 0
        prim_router.Send_PI_en = 0
        prim_router.Back_sign_en = 0
        prim_router.Send_PI_num = 0
        prim_router.Receive_sign_num = 0
        prim_router.Send_PI_addr_base = 0
        prim_router.Relay_number = 1
        prim_router.Q = 0
        prim_router.T_mode = 0
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 0

        prim_router.recv_source_core_grp.append({'core_id': ((1,0),(idx, idy)), 'data_num': 896,'T_mode': 1, 'Rhead_num': 1})

        # prim_in = prim_router.init_data()
        # prim_router.memory_blocks = [
        #     {'name': 'Router_receive',
        #         'start': 0x8400,
        #         'length': 3584,  #224*16
        #         'data':  prim_in[0],
        #         'mode': 0},
        # ]

        #  move prim in 1-4 phases of core1
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 56*32
        prim_soma2.length_out = 56*32
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 4
        prim_soma2.num_out = 4
        prim_soma2.num_ciso = 0
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0 
        prim_soma2.Addr_Start_in = 0x8400     # 21000 
        prim_soma2.Addr_Start_out = start_out     
        prim_soma2.Addr_Start_ciso = 0x79A8   # 1E6A0
        prim_soma2.in_row_max = 0 
        prim2_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Router_receive',
                'start': 0x8400,
                # 'length': 3584,    #  224*16
                'data':  prim2_in[0],
                'mode': 0,
                'initialize': False
                },
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]

        return prim_router, prim_soma2


    def config_prim_sending_c1(self):
        for phase in range(self.sync_num):   
            if (phase+1) == 1:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0000, batch=1) 
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_sr(batch=1,x=0,y=-2, Addr_Rhead_base=0x300, idx=0, idy=0, idrx=15, idry=1, phase=phase, move_in=0x2000) 

                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch1', 0)]
                one_phase_dict['router'] = [prim_router] #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)              
            
            elif (phase+1) == 2:
                one_phase_dict = {} 
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0E00, batch=2)  
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_sr(batch=2,x=4,y=-2, Addr_Rhead_base=0x308, idx=4, idy=0, idrx=15, idry=1, phase=phase, move_in=0x2700) 
                # phase dict  
                one_phase_dict['axon'] = [prim_axon]  
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch2', 0)]
                one_phase_dict['router'] = [prim_router] #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 3:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x4000, batch=3)  # 原：1C00
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_sr(batch=3,x=8,y=-2, Addr_Rhead_base=0x310, idx=8, idy=0, idrx=15, idry=1, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch3', 0)]
                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 4:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x4E00, batch=4)  # 原：2A00
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_sr(batch=4,x=12,y=-2, Addr_Rhead_base=0x318, idx=12, idy=0, idrx=15, idry=1, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch4', 0)]
                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 5:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=5, Addr_Rhead_base=0x320, idrx=15, idry=1, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)
            
            elif (phase+1) == 6:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=6, Addr_Rhead_base=0x328, idrx=15, idry=1, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 7:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=7, Addr_Rhead_base=0x330, idrx=15, idry=1, phase=phase, move_in=0x7100)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 22:
                one_phase_dict = {}
                # move
                prim_soma1 = self.prim_pingpang(addr_in=0x2000, addr_out=0x0000,num=32*2)
                # phase dict
                one_phase_dict['axon'] = None # [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1]
                one_phase_dict['router'] = None  
                one_phase_dict['soma2'] = None
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 23:
                one_phase_dict = {}
                # move
                prim_soma1 = self.prim_pingpang(addr_in=0x5500, addr_out=0x0E00,num=32*2)
                # phase dict
                one_phase_dict['axon'] = None # [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1]
                one_phase_dict['router'] = None  
                one_phase_dict['soma2'] = None
                self.prim_list_c1.append(one_phase_dict)

            elif (phase+1) == 24:
                one_phase_dict = {}
                # move
                prim_soma1 = self.prim_pingpang(addr_in=0x6300, addr_out=0x4000,num=32*3)
                # phase dict
                one_phase_dict['axon'] = None # [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1]
                one_phase_dict['router'] = None  
                one_phase_dict['soma2'] = None
                self.prim_list_c1.append(one_phase_dict)

            else:
                self.prim_list_c1.append(self.none_phase())

            # return self.prim_list_c1

    def none_phase(self):
        one_phase_dict = {}
        # phase dict
        one_phase_dict['axon'] = None
        one_phase_dict['soma1'] = None
        one_phase_dict['router'] = None
        one_phase_dict['soma2'] = None

        return one_phase_dict

            

    def config_sending_c1(self):  # sendding core1
        self.config_prim_sending_c1()
        sending1_cofig = {  # core id
            'memory_blocks': {
                'sending_batch1': {'start': 0x0000, 'length': 224*16*4, 'initialize': False},
                'sending_batch2': {'start': 0x0E00, 'length': 224*16*4, 'initialize': False},
                'sending_batch3': {'start': 0x1C00, 'length': 224*16*4, 'initialize': False},
                'sending_batch4': {'start': 0x2A00, 'length': 224*16*2, 'initialize': False},
                'router_buff': {'start': 0x8400, 'length': 224*16*1, 'initialize': False},

                'Receive_batch1': {'start': 0x4000, 'length': 224*16*2, 'initialize': False},
                'Receive_batch2': {'start': 0x4700, 'length': 224*16*2, 'initialize': False},
                'Receive_batch3': {'start': 0x4E00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch4': {'start': 0x5500, 'length': 224*16*2, 'initialize': False},
                'Receive_batch5': {'start': 0x5C00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch6': {'start': 0x6300, 'length': 224*16*2, 'initialize': False},
                'Receive_batch7': {'start': 0x6A00, 'length': 224*16*2, 'initialize': False},
                'pingpang_move': {'start': 0x4000, 'length': 224*224, 'initialize': False},
            },
            'prims': self.prim_list_c1
        }
        return sending1_cofig


    def prim_move_sending(self, start_in, batch, spe=4):      
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 14*16
        prim_soma2.length_out = 14*16
        prim_soma2.length_ciso = 0
        
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = start_in   
        prim_soma2.Addr_Start_out = 0x9000     
        prim_soma2.Addr_Start_ciso = 0x4000 
        prim_soma2.in_row_max = 0
        prim_soma2.mem_sel = 1
        if batch == spe:
            length = 224*16*2
            prim_soma2.num_in = 16*2
            prim_soma2.num_out = 16*2
            prim_soma2.num_ciso = 16*2
        else:
            length = 224*16*4
            prim_soma2.num_in = 16*4
            prim_soma2.num_out = 16*4
            prim_soma2.num_ciso = 16*4
        
        prim2_in = prim_soma2.init_data()

        prim_in = []
        for aaa in range(len(prim2_in[0])):  #  // 4
            # tmp=[]  
            # for bbb in range(4):
            #     tmp.append(1)
            prim_in.append([8])

        prim_soma2.memory_blocks = [
            {'name': 'sending_batch{}'.format(batch),
                'start': start_in,
                # 'length': length,
                'data':  prim_in,
                'mode': 0,
                'initialize': True},
            # {'name': 'Input_buff_RGB',
            #     'start': start_out,
            #     'length': 3584,
            #     'data': prim2_in[1],
            #     'mode': 0},
        ]

        return prim_soma2

    def prim_router_sending(self,batch, x, y, Addr_Rhead_base, idx, idy, phase, spe=4):
        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 0
        # prim_router.Dout_Mem_sel = 1   # 2020.7.17修改，// 0
        if batch == spe:
            prim_router.Send_number = 895  # 发送的所有包数，包含EN=0的包，需要减1
           
        else:
            prim_router.Send_number = 1791
            
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 21000
        prim_router.Addr_Dout_length = 223  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 1  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x1000  # 4B寻址 21000
        prim_router.Addr_Din_length = 0  # 需要减1
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1

        prim_router.Dout_Mem_sel = 1


        if batch == spe:
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx, idy)),((1,0),(idx, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+1, idy)),((1,0),(idx+1, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
        else:
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx, idy)),((1,0),(idx, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+1, idy)),((1,0),(idx+1, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+2, idy)),((1,0),(idx+2, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+3, idy)),((1,0),(idx+3, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式

        return prim_router

    def prim_router_sr(self,batch, x, y, Addr_Rhead_base, idx, idy, idrx, idry, phase, move_in, spe=4):
        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 1
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 1
        if batch == spe:
            prim_router.Send_number = 895  # 发送的所有包数，包含EN=0的包，需要减1
           
        else:
            prim_router.Send_number = 1791
            
        prim_router.Addr_Dout_base = 0x1000  # 4B寻址 21000
        prim_router.Addr_Dout_length = 223  # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 1  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 895  # 需要减1 224*32/8
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
        prim_router.T_mode = 1
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 1
        
        prim_router.recv_source_core_grp.append({'core_id': ((0,0), (idrx,idry)), 'data_num': 896,'T_mode': 1, 'Rhead_num': 1, 'sync_en':1, 'sync_phase_num': phase})

        if batch == spe:
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx, idy)),((1,0),(idx, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+1, idy)),((1,0),(idx+1, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
        else:
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx, idy)),((1,0),(idx, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+1, idy)),((1,0),(idx+1, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+2, idy)),((1,0),(idx+2, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})
            prim_router.send_destin_core_grp.append({'core_id': [((1,0),(idx+3, idy)),((1,0),(idx+3, idy+1))], 'data_num': 448,'T_mode': 1, 'Rhead_num': 1,'sync_en':1, 'sync_phase_num': phase})

            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式
            x += 1
            prim_router.addRHead(S=0, T=1, P=0, Q=1, X=x, Y=y, A=0, pack_per_Rhead=447, A_offset=0, Const=447, EN=1) # 一对一包头参数形式

        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 224
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 32
        prim_soma2.num_ciso = 32
        prim_soma2.length_out = 224
        prim_soma2.num_out = 32
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma2.Addr_Start_out = move_in    
        prim_soma2.in_row_max = 0
        prim_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Receive_batch{}'.format(batch),
                'start': prim_soma2.Addr_Start_in,
                # 'length': 672,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
        ]

        return prim_router, prim_soma2
    

    def prim_router_r(self,batch, Addr_Rhead_base, idrx, idry, phase, move_in, spe=4):
        # 0x09 router
        prim_router = Prim_09_Router()
        prim_router.Rhead_mode = 1
        prim_router.CXY = 0b00
        prim_router.Send_en = 0
        prim_router.Receive_en = 1
        prim_router.Dout_Mem_sel = 0
        prim_router.Send_number = 0  # 发送的所有包数，包含EN=0的包，需要减1
           
       
            
        prim_router.Addr_Dout_base = 0  # 4B寻址 21000
        prim_router.Addr_Dout_length = 0 # 16B的个数，需要减1
        prim_router.Addr_Rhead_base = Addr_Rhead_base  # 4B寻址 20C00
        prim_router.Addr_Rhead_length = 1  # 16B的个数，需要减1
        prim_router.Addr_Din_base = 0x400  # 4B寻址 21000
        prim_router.Addr_Din_length = 895  # 需要减1 224*32/8
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
        prim_router.T_mode = 0
        prim_router.Receive_sign_en = 0
        prim_router.Soma_in_en = 0
        
        prim_router.recv_source_core_grp.append({'core_id': ((0,0), (idrx,idry)), 'data_num': 896,'T_mode': 1, 'Rhead_num': 1, 'sync_en':1, 'sync_phase_num': phase})

        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 224
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = 32
        prim_soma2.num_ciso = 32
        prim_soma2.length_out = 224
        prim_soma2.num_out = 32
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = 0x8400   # 21000
        prim_soma2.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma2.Addr_Start_out = move_in    #  0x4000  # 10000
        prim_soma2.in_row_max = 0
        prim_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'Receive_batch{}'.format(batch),
                'start': prim_soma2.Addr_Start_in,
                # 'length': 672,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
        ]

        return prim_router, prim_soma2

    def prim_pingpang(self, addr_in=0x4000, addr_out=0x0000, num=224):
        # 0x06 move
        prim_soma2 = Prim_06_move_merge()
        prim_soma2.length_in = 224
        prim_soma2.length_ciso = 0
        prim_soma2.num_in = num
        prim_soma2.num_ciso = num
        prim_soma2.length_out = 224
        prim_soma2.num_out = num
        prim_soma2.type_in = 1
        prim_soma2.type_out = 1
        prim_soma2.in_cut_start = 0
        prim_soma2.Reset_Addr_in = 1
        prim_soma2.Reset_Addr_out = 1
        prim_soma2.Reset_Addr_ciso = 1
        prim_soma2.Row_ck_on = 0
        prim_soma2.Addr_Start_in = addr_in   
        prim_soma2.Addr_Start_ciso = 0x79A8  # 1E6A0
        prim_soma2.Addr_Start_out = addr_out    
        prim_soma2.in_row_max = 0
        prim_in = prim_soma2.init_data()
        prim_soma2.memory_blocks = [
            {'name': 'pingpang_move',
                'start': prim_soma2.Addr_Start_in,
                # 'length': 672,
                'data':  prim_in[0],
                'mode': 0,
                'initialize': False},
        ]

        return prim_soma2


    def config_prim_sending_c2(self):
        for phase in range(self.sync_num):   
            if (phase+1) == 1:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0000, batch=1, spe=3) 
                # 0x09 router
                prim_router = self.prim_router_sending(batch=1,x=4,y=-3, Addr_Rhead_base=0x300, idx=4, idy=0, phase=phase, spe=3) 

                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch1', 0)]
                one_phase_dict['router'] = [prim_router] #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c2.append(one_phase_dict)              
            
            elif (phase+1) == 2:
                one_phase_dict = {} 
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0E00, batch=2, spe=3)  
                # 0x09 router
                prim_router = self.prim_router_sending(batch=2,x=8,y=-3, Addr_Rhead_base=0x308, idx=8, idy=0, phase=phase,spe=3) 
                # phase dict  
                one_phase_dict['axon'] = [prim_axon]  
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch2', 0)]
                one_phase_dict['router'] = [prim_router] #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 3:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x1C00, batch=3, spe=3)
                # 0x09 router
                prim_router = self.prim_router_sending(batch=3,x=12,y=-3, Addr_Rhead_base=0x310, idx=12, idy=0, phase=phase, spe=3)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch3', 0)]
                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 4:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x2300, batch=4, spe=3)
                # 0x09 router
                prim_router = self.prim_router_sending(batch=4,x=0,y=-3, Addr_Rhead_base=0x318, idx=0, idy=0, phase=phase, spe=3)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch3', 0)]
                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 8:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=8, Addr_Rhead_base=0x320, idrx=15, idry=2, phase=phase, move_in=0x4000)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 9:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=9, Addr_Rhead_base=0x324, idrx=15, idry=2, phase=phase, move_in=0x4700)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 10:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x328, idrx=15, idry=2, phase=phase, move_in=0x4E00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict)    

            elif (phase+1) == 11:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x32C, idrx=15, idry=2, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict) 

            elif (phase+1) == 12:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x330, idrx=15, idry=2, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict) 

            elif (phase+1) == 13:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x334, idrx=15, idry=2, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 14:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x338, idrx=15, idry=2, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c2.append(one_phase_dict)

            elif (phase+1) == 22:
                one_phase_dict = {}
                # move
                prim_soma1 = self.prim_pingpang()
                # phase dict
                one_phase_dict['axon'] = None # [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1]
                one_phase_dict['router'] = None  
                one_phase_dict['soma2'] = None
                self.prim_list_c2.append(one_phase_dict)
            
            else:
                self.prim_list_c2.append(self.none_phase())

            # return self.prim_list_c2


    def config_sending_c2(self):  # sendding core1
        self.config_prim_sending_c2()
        sending2_cofig = {  # core id
            'memory_blocks': {
                'sending_batch1': {'start': 0x0000, 'length': 224*16*4, 'initialize': False},
                'sending_batch2': {'start': 0x0E00, 'length': 224*16*4, 'initialize': False},
                'sending_batch3': {'start': 0x1C00, 'length': 224*16*2, 'initialize': False},
                'sending_batch4': {'start': 0x2300, 'length': 224*16*4, 'initialize': False},
                'router_buff': {'start': 0x8400, 'length': 224*16*1, 'initialize': False},

                'Receive_batch8': {'start': 0x4000, 'length': 224*16*2, 'initialize': False},
                'Receive_batch9': {'start': 0x4700, 'length': 224*16*2, 'initialize': False},
                'Receive_batch10': {'start': 0x4E00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch11': {'start': 0x5500, 'length': 224*16*2, 'initialize': False},
                'Receive_batch12': {'start': 0x5C00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch13': {'start': 0x6300, 'length': 224*16*2, 'initialize': False},
                'Receive_batch14': {'start': 0x6A00, 'length': 224*16*2, 'initialize': False},
                'pingpang_move': {'start': 0x4000, 'length': 224*224, 'initialize': False},
            },
            'prims': self.prim_list_c2
        }
        return sending2_cofig

    
    def config_prim_sending_c3(self):
        for phase in range(self.sync_num):  
            if (phase+1) == 1:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0000, batch=1, spe=2) 
                # 0x09 router
                prim_router = self.prim_router_sending(batch=1,x=8,y=-4, Addr_Rhead_base=0x300, idx=8, idy=0, phase=phase, spe=2) 

                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch1', 0)]
                one_phase_dict['router'] = [prim_router]  #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c3.append(one_phase_dict)              
            
            elif (phase+1) == 2:
                one_phase_dict = {} 
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x0E00, batch=2, spe=2)  
                # 0x09 router
                prim_router = self.prim_router_sending(batch=2,x=12,y=-4, Addr_Rhead_base=0x308, idx=12, idy=0, phase=phase, spe=2) 
                # phase dict  
                one_phase_dict['axon'] = [prim_axon]  
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch2', 0)]
                one_phase_dict['router'] = [prim_router] #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 3:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x1500, batch=3, spe=2)
                # 0x09 router
                prim_router = self.prim_router_sending(batch=3,x=0,y=-4, Addr_Rhead_base=0x310, idx=0, idy=0, phase=phase, spe=2)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch3', 0)]

                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 4:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81()
                # 0x06 merge 
                prim_soma1 = self.prim_move_sending(start_in=0x2300, batch=4, spe=2)
                # 0x09 router
                prim_router = self.prim_router_sending(batch=4,x=4,y=-4, Addr_Rhead_base=0x318, idx=4, idy=0, phase=phase, spe=2)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1, ('sending_batch3', 0)]
                one_phase_dict['router'] = [prim_router]   #  ,  ('router_buff', 0)
                one_phase_dict['soma2'] = None
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 15:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x320, idrx=15, idry=3, phase=phase, move_in=0x4000)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 16:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x324, idrx=15, idry=3, phase=phase, move_in=0x4700)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 17:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x328, idrx=15, idry=3, phase=phase, move_in=0x4E00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict)    

            elif (phase+1) == 18:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x32C, idrx=15, idry=3, phase=phase, move_in=0x5500)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict) 

            elif (phase+1) == 19:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x330, idrx=15, idry=3, phase=phase, move_in=0x5C00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict) 

            elif (phase+1) == 20:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x334, idrx=15, idry=3, phase=phase, move_in=0x6300)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 21:
                one_phase_dict = {}
                # auto cnn81
                prim_axon = self.auto_cnn81(addr=0x3E00)
                # 0x09 router
                prim_router, prim_soma2 = self.prim_router_r(batch=phase+1, Addr_Rhead_base=0x338, idrx=15, idry=3, phase=phase, move_in=0x6A00)
                # phase dict
                one_phase_dict['axon'] = [prim_axon]
                one_phase_dict['soma1'] = None
                one_phase_dict['router'] = [prim_router]   
                one_phase_dict['soma2'] = [prim_soma2]
                self.prim_list_c3.append(one_phase_dict)

            elif (phase+1) == 22:
                one_phase_dict = {}
                # move
                prim_soma1 = self.prim_pingpang()
                # phase dict
                one_phase_dict['axon'] = None #  [prim_axon]
                one_phase_dict['soma1'] = [prim_soma1]
                one_phase_dict['router'] = None  
                one_phase_dict['soma2'] = None
                self.prim_list_c3.append(one_phase_dict)

            else:
                self.prim_list_c3.append(self.none_phase())

            # return self.prim_list_c1

    def config_sending_c3(self):  # sendding core1
        self.config_prim_sending_c3()
        sending3_cofig = {  # core id
            'memory_blocks': {
                'sending_batch1': {'start': 0x0000, 'length': 224*16*4, 'initialize': False},
                'sending_batch2': {'start': 0x0E00, 'length': 224*16*2, 'initialize': False},
                'sending_batch3': {'start': 0x1500, 'length': 224*16*4, 'initialize': False},
                'sending_batch4': {'start': 0x2300, 'length': 224*16*4, 'initialize': False},
                'router_buff': {'start': 0x8400, 'length': 224*16*1, 'initialize': False},

                'Receive_batch15': {'start': 0x4000, 'length': 224*16*2, 'initialize': False},
                'Receive_batch16': {'start': 0x4700, 'length': 224*16*2, 'initialize': False},
                'Receive_batch17': {'start': 0x4E00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch18': {'start': 0x5500, 'length': 224*16*2, 'initialize': False},
                'Receive_batch19': {'start': 0x5C00, 'length': 224*16*2, 'initialize': False},
                'Receive_batch20': {'start': 0x6300, 'length': 224*16*2, 'initialize': False},
                'Receive_batch21': {'start': 0x6A00, 'length': 224*16*2, 'initialize': False},
                'pingpang_move': {'start': 0x4000, 'length': 224*224, 'initialize': False},
            },
            'prims': self.prim_list_c3
        }
        return sending3_cofig


#     def temp_ir_group1(self):  #  IR for group1
#         group_config = {}
#         # group_config_temp = {}
#         group_config['clock'] = 70000
#         group_config['mode'] = 0
#         core1_config = self.config_core1()
#         group_config[((0,0),(0, 1))] = core1_config
#         for i in range(3):   #  3
#             for j in range(1, 3):  # 2
#                 if (i, j) == (0, 1):
#                     continue
#                 group_config[((0,0), (i, j))] = self.config_core_from_core1(
#                     core1_config, i, j)      
#         return group_config[((1, 0), (0, 1))]

#     def temp_ir_source(self):  # IR for sending cores (16,0),(17,0),(16,1)
#         group_config = {}
#         group_config['clock'] = 6000
#         group_config['mode'] = 0
#         group_config[((1, 0), (0, 0))] = self.config_sending_c1()
#         group_config[((1, 0), (0, 1))] = self.config_sending_c2()
#         group_config[((1, 0), (0, 2))] = self.config_sending_c3()
        
#         return group_config[((1, 0), (0, 2))]

    
#     def temp_check(self):
#         # sending3_cofig = self.config_sending_c3()
#         temp = self.temp_ir_group1()
#         temp2 = self.temp_ir_source()
#         print('*******temp**checkout********')
#         weight = temp['prims'][10]['axon']
#         print('w:',  weight) 
#         print('w:',  weight[0].memory_blocks[1]['data']) 
#         print('lenth:',  len(weight[0].memory_blocks[1]['data'])) 
#         # print(self.config_prim_sending_c1())
#         # for phase in range(20):
#         #     print('--{}---'.format(phase), temp['prims'][phase]) 
        

# LogicalIRGenerator().temp_check()   #  check what you want check
# _, gg = LogicalIRGenerator().format_ir_group1()
# print('***14,0****', gg['prims'])


    



        

           