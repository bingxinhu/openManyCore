
from primitive import Prim_41_Axon, Prim_X5_Soma, Prim_09_Router, Prim_06_move_merge, Prim_06_move_split
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.code_generator import MapConfig, GroupConfig

SIMPATH = '..\\..\\Out_files\\Axon-Soma1-Router-Soma2\\'

np.random.seed(0x999)

# out   :0~ff,200~2ff
# ciso  :100~1ff,300~3ff
Soma_Prim = Prim_06_move_split()
Soma_Prim.length_in = 512
Soma_Prim.num_in = 2
Soma_Prim.length_out = 256
Soma_Prim.length_ciso = 256
Soma_Prim.num_out = 2
Soma_Prim.num_ciso = 2
Soma_Prim.type_in = 0
Soma_Prim.type_out = 0
Soma_Prim.in_cut_start = 0
Soma_Prim.Reset_Addr_in = 1
Soma_Prim.Reset_Addr_out = 1
Soma_Prim.Reset_Addr_ciso = 1
Soma_Prim.Row_ck_on = 0
Soma_Prim.Addr_Start_in = 0x8400
Soma_Prim.Addr_Start_ciso = 0x4000
Soma_Prim.Addr_Start_out = 0x9000
Soma_Prim.in_row_max = 0
Soma_Prim.mem_sel = 1
Soma_Prim.out_ciso_sel = 0
data = []
for i in range(1024):
    data.append([i])
Soma_Prim.memory_blocks = [
    {
        "name": "test_data",
        "start": Soma_Prim.Addr_Start_in,
        "data": data}
]

Router_Prim = Prim_09_Router()
Router_Prim.Rhead_mode = 0
Router_Prim.CXY = 0
Router_Prim.Send_en = 1
Router_Prim.Receive_en = 1
Router_Prim.Dout_Mem_sel = 1
Router_Prim.Send_number = 255  # 发送的所有包数，包含EN=0的包，需要减1
Router_Prim.Addr_Dout_base = 0x1000  # 4B寻址
Router_Prim.Addr_Dout_length = 63  # 16B的个数，需要减1
Router_Prim.Addr_Rhead_base = 0x300  # 4B寻址
Router_Prim.Addr_Rhead_length = 63  # 16B的个数，需要减1
Router_Prim.Addr_Din_base = 0x800  # 4B寻址
Router_Prim.Addr_Din_length = 255  # 需要减1
Router_Prim.Receive_number = 255
Router_Prim.Nx = 0
Router_Prim.Ny = 0
Router_Prim.Send_PI_en = 0
Router_Prim.Back_sign_en = 0
Router_Prim.Send_PI_num = 0
Router_Prim.Receive_sign_num = 0
Router_Prim.Send_PI_addr_base = 0
Router_Prim.Relay_number = 0
Router_Prim.Q = 0
Router_Prim.T_mode = 1
Router_Prim.Receive_sign_en = 0
Router_Prim.Soma_in_en = 1


for i in range(256):
    Router_Prim.addRHead(S=0, T=1, P=1, Q=0, X=0, Y=0, A=i)
    pass

# Router_Prim.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0,
#                      pack_per_Rhead=255, A_offset=0, Const=0, EN=1)


# 映射策略\
map_config = {
    'sim_clock': 5000,
    0: {
        0: {'clock': 5000,
            'mode': 1,
            ((0, 0), (0, 0)): {
                'prims': [{'axon': None, 'soma1': Soma_Prim, 'router': Router_Prim, 'soma2': None}]
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


# 测试配置
test_config = {
    'tb_name': "test_S1R_mem2_wr",
    'test_mode': TestMode.MEMORY_STATE,
    'test_group_phase': test_phase  # (phase_group, phase_num)
}


# 开始测试


def test_case():
    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
