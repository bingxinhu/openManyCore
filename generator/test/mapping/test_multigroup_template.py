import pytest
import numpy as np
import os

from primitive import Prim_41_Axon, Prim_X5_Soma
from generator.test_engine import TestMode, TestEngine
SIMPATH = 'temp\\out_files\\'
tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
np.random.seed(sum(ord(c) for c in tb_name))

# 41和05流水，配置两个组，每个组两个core

Axon_Prim_1 = Prim_41_Axon()
Axon_Prim_1.pad_on = False
Axon_Prim_1.InA_type = 1
Axon_Prim_1.InB_type = 1
Axon_Prim_1.Load_Bias = 0
Axon_Prim_1.cin = 16
Axon_Prim_1.cout = 32
Axon_Prim_1.Input_fm_Px = 10
Axon_Prim_1.Input_fm_Py = 10
Axon_Prim_1.pad_up = 0
Axon_Prim_1.pad_down = 0
Axon_Prim_1.pad_left = 0
Axon_Prim_1.pad_right = 0
Axon_Prim_1.conv_Kx = 3
Axon_Prim_1.conv_Ky = 3
Axon_Prim_1.conv_Sx = 1
Axon_Prim_1.conv_Sy = 1
Axon_Prim_1.conv_Ex = 1
Axon_Prim_1.conv_Ey = 1

Axon_Prim_1.Addr_InA_base = 0x0000
Axon_Prim_1.Addr_InB_base = 0x4000
Axon_Prim_1.Addr_V_base = 0x2000

a = Axon_Prim_1.init_data()
Axon_Prim_1.memory_blocks = [
    {'name': "input_X",
     'start': Axon_Prim_1.Addr_InA_base,
     'data': a[0],
     'mode': 0},
    {'name': "weight",
     'start': Axon_Prim_1.Addr_InB_base,
     'data': a[1],
     'mode': 0}
]

Soma_Prim = Prim_X5_Soma()
Soma_Prim.pad_on = False
Soma_Prim.CMP_C_en = False
Soma_Prim.type_in = 0
Soma_Prim.type_out = 0
Soma_Prim.cin = 32  # 为支持channel数不足16B的情况，所以用户可直接设置cin，程序会自动计算X_Km_num
Soma_Prim.cout = 32
Soma_Prim.Input_fm_Px = 8
Soma_Prim.Input_fm_Py = 8
Soma_Prim.pad_top = 0
Soma_Prim.pad_down = 0
Soma_Prim.pad_left = 0
Soma_Prim.pad_right = 0
Soma_Prim.pooling_Kx = 2
Soma_Prim.pooling_Ky = 2
Soma_Prim.pooling_Sx = 2
Soma_Prim.pooling_Sy = 2
Soma_Prim.CMP_C = 0x80000000
Soma_Prim.in_cut_start = 12
Soma_Prim.in_row_max = 2

Soma_Prim.reset_Addr_in = 1
Soma_Prim.reset_Addr_out = 1
Soma_Prim.Addr_Start_in = 0x2000
Soma_Prim.Addr_Start_out = 0x6000
Soma_Prim.Row_ck_on = 1

# 映射策略\
map_config = {
    'sim_clock':8000
    0: {
        0: {'clock': 8000,
            'mode':1,
            ((0, 0), (0, 0)): {
                    'prims':[{'axon':Axon_Prim_1,'soma1':Soma_Prim,'router':None,'soma2':None}]
                    'instant_prims':[{'axon':None,'soma1':None,'router':None,'soma2':None}]
            },
        1:  {'clock': 3000, 
            'mode':1,
            ((0, 0), (1, 0)): {
                    'prims':[{'axon':Axon_Prim_1,'soma1':Soma_Prim,'router':None,'soma2':None}]
                    'instant_prims':[{'axon':None,'soma1':None,'router':None,'soma2':None}]
                }
            }
        },
    }
}

# 测试配置
test_config = {
    'tb_name': tb_name,
    'test_mode': TestMode.MEMORY_STATE,
    'test_group_phase': [(0, 1),(1,1)]    # (phase_group, phase_num)
}

# 开始测试
def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()

if __name__ =="__main__":
    test_case()