import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G18_IB(phase_en, clock, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group18 Input Buffer
    1 cores
    chip(1, 0)
    (15, 9)
    
    """
    map_config = {
        'sim_clock': None,
            0:{
                'clock':None,
                0:{
                    'clock': clock,
                    'trigger':0,
                    'mode': 1
                }
            }
        }
    for (core_x, core_y) in [(15, 0)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    """
    接收18*512
    """
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(15, 0)]:
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=1599, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                #*******************************
                router.Receive_number = 15
                router.Addr_Din_length = 255
                #*******************************
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=2048, \
                            cout=2048, px=1, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    return map_config


"""# ---------------------------------------------------------------------------------------#
# 真正执行的代码
phase_en = np.zeros(32)
# 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
# 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
# group间
phase_en[0] = 0     # 组间
phase_en[1] = 0     # 组间

phase_en[2] = 1     # 44卷积计算，部分和收发
phase_en[3] = 1     # 44部分和求和
phase_en[24] = 1    # 44多播
phase_en[4] = 1     # X1数据整理成Xe

phase_en[5] = 1     # 45层卷积计算   --- 时钟较大 2.2k可以满足
phase_en[6] = 1     # 45部分和求和
phase_en[25] = 1    # 45层多播

phase_en[8] = 1     # 46层卷积，部分和收发 ---1
phase_en[9] = 1     # 46层部分和求和，流水截取 ---1
phase_en[10] = 1    # 46层卷积，部分和收发 ---2
phase_en[11] = 1    # 46层部分和求和，流水截取 ---2
phase_en[12] = 1    # 46层输出与Xe相加，输出保存在Mem1 0x19000

run = True  # 只生成map_config 还是生成后直接运行

map_config = Gen_G16_Map_Config(phase_en, 22000, M=16, N=5)

import pickle
with open('G16_64cores\\G16_64cores_phase_1_12', 'wb') as f:
    pickle.dump(map_config, f)

if run:
    from generator.test_engine import TestMode, TestEngine

    test_phase = []
    for i in range(len(map_config[0][0][((0, 0), (0, 0))]['axon'])):
        test_phase.append((0, i + 1))
    map_config['sim_clock'] = min(len(map_config[0][0][((0, 0), (0, 0))]['axon']) * map_config[0][0]['clock'] - 1, 100000)
    test_config = {
            'tb_name': 'M99999',
            'test_mode': TestMode.MEMORY_STATE,
            'test_group_phase': test_phase
        }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()"""