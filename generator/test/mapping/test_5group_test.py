import os
import numpy as np


def test_case():
    import generator.functions.data_generator as data_generator
    from primitive.Prim_X5_Soma_compare_new import Prim_X5_Soma
    from primitive.Prim_41_Axon_CNN0_new import Prim_41_Axon
    from generator.test_engine import TestMode, TestEngine

    SIMPATH = 'temp\\out_files\\'
    tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
    np.random.seed(sum(ord(c) for c in tb_name))

    Axon_Prim = Prim_41_Axon()
    Axon_Prim.pad_on = False
    Axon_Prim.InA_type = 1  # C++仿真器不支持InA_type = 3
    Axon_Prim.InB_type = 1
    Axon_Prim.Load_Bias = 0
    Axon_Prim.cin = 32
    Axon_Prim.cout = 32
    Axon_Prim.Input_fm_Px = 1
    Axon_Prim.Input_fm_Py = 1
    Axon_Prim.pad_up = 0
    Axon_Prim.pad_down = 0
    Axon_Prim.pad_left = 0
    Axon_Prim.pad_right = 0
    Axon_Prim.conv_Kx = 1
    Axon_Prim.conv_Ky = 1
    Axon_Prim.conv_Sx = 1
    Axon_Prim.conv_Sy = 1
    Axon_Prim.conv_Ex = 1
    Axon_Prim.conv_Ey = 1

    Axon_Prim.Reset_Addr_A = 1
    Axon_Prim.Reset_Addr_V = 1
    Axon_Prim.Addr_InA_base = 0x4000  # 0x10000
    Axon_Prim.Addr_InB_base = 0x0000
    Axon_Prim.Addr_V_base = 0x2000  # 0x156A0
    a = Axon_Prim.init_data()
    Axon_Prim.memory_blocks = [
        {'name': "intput_x",
         'start': Axon_Prim.Addr_InA_base,
         'data': a[0],
         'mode': 0},
         {'name': "intput_w",
         'start': Axon_Prim.Addr_InB_base,
         'data': a[1],
         'mode': 0}
    ]
    map_config = {
        0: {
            0: {'clock': 2000,                # phase group id
                # 'mode': PhaseMode.SELF_ADAPT,
                ((0, 0), (0, 0)): {
                    'axon': [Axon_Prim],    # 测试在代码生成器之上时：任务性原语，反之：执行性原语
                    'soma1': [None],
                    'router': [None],
                    'soma2': [None]
                }
            },
            1: {'clock': 2000,                # phase group id
                # 'mode': PhaseMode.SELF_ADAPT,
                ((0, 0), (0, 1)): {
                    'axon': [Axon_Prim],    # 测试在代码生成器之上时：任务性原语，反之：执行性原语
                    'soma1': [None],
                    'router': [None],
                    'soma2': [None]
                }
            },
            2: {'clock': 2000,                # phase group id
                # 'mode': PhaseMode.SELF_ADAPT,
                ((0, 0), (0, 2)): {
                    'axon': [Axon_Prim],    # 测试在代码生成器之上时：任务性原语，反之：执行性原语
                    'soma1': [None],
                    'router': [None],
                    'soma2': [None]
                }
            },
            3: {'clock': 2000,                # phase group id
                # 'mode': PhaseMode.SELF_ADAPT,
                ((0, 0), (0, 3)): {
                    'axon': [Axon_Prim],    # 测试在代码生成器之上时：任务性原语，反之：执行性原语
                    'soma1': [None],
                    'router': [None],
                    'soma2': [None]
                }
            },
            4: {'clock': 2000,                # phase group id
                # 'mode': PhaseMode.SELF_ADAPT,
                ((0, 0), (0, 4)): {
                    'axon': [Axon_Prim],    # 测试在代码生成器之上时：任务性原语，反之：执行性原语
                    'soma1': [None],
                    'router': [None],
                    'soma2': [None]
                }
            },
        }
    }

    # 测试配置
    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'test_group_phase': [(0, 1),(1, 1),(2, 1),(3, 1),(4, 1)]    # (phase_group, phase_num)
    }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    test_case()
