from generator.test.SCH_Mapping.G15_64cores import Gen_G15_Map_Config
from generator.test.SCH_Mapping.G16_64cores import Gen_G16_Map_Config
from generator.test.SCH_Mapping.G17_64cores import Gen_G17_Map_Config
from generator.test.SCH_Mapping.G14_32cores_4_8 import Gen_G14_Map_Config
from generator.test.SCH_Mapping.G16_IB import Gen_G16_IB
from generator.test.SCH_Mapping.G15_OB import Gen_G15_OB
from generator.test.SCH_Mapping.AddRouterInfo import add_router_info
from generator.test.SCH_Mapping.G13_32cores_4_8 import Gen_G13_Map_Config
from generator.test.SCH_Mapping.G13_IB_for_4_8 import Gen_G13_IB
from generator.test.SCH_Mapping.G12_32cores_4_8 import Gen_G12_Map_Config
from generator.test.SCH_Mapping.G12_OB_for_4_8 import Gen_G12_OB
from generator.test.SCH_Mapping.G11_32cores_4_8 import Gen_G11_Map_Config
from generator.test.SCH_Mapping.G10_32cores_4_8 import Gen_G10_Map_Config
from generator.test.SCH_Mapping.G9_48cores_6_8 import Gen_G9_Map_Config
from generator.test.SCH_Mapping.G9_IB_for_6_8 import Gen_G9_IB
from generator.test.SCH_Mapping.G18_IB import Gen_G18_IB
from generator.test.SCH_Mapping.G12_G13_relay_core import Gen12_Gen13_relay_cores
from generator.test.SCH_Mapping.changeIR_to_prims import change_ir_to_prims
import numpy as np
import os

case_file_name = 'M99998'
c_path = os.getcwd()
out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
if os.path.exists(out_files_path):
    os.chdir(out_files_path)
    del_command = 'rd/s/q cmp_out'
    os.system(del_command)
    os.chdir(c_path)

map_config = {
        'sim_clock': None,
        0: {                     # step group id
                'clock': None,

            }
        }
for i in range(50):
    map_config[0][i] = {
        'clock': 80000,
        'trigger': 0,
        'mode': 1
    }

save = False            # 是否保存mapping ir文件
run = True              # 是否运行仿真器
self_adopt = True       # 是否开启自适应phase
debug_top = False        # 是否打印debug top信息
save_LUT = False        # 是否单独保存路由表
group_idx_list = []


# 每个phase group单独的使能开关，0表示忽略该phase group
G9_IB_en = 1
G9_en = 1
G10_en = 1
G11_en = 1
G12_en = 1
G12_OB_en = 1

G13_IB_en = 1
G13_en = 1
G14_en = 1
G15_en = 1
G15_OB_en = 1

G16_IB_en = 1
G16_en = 1
G17_en = 1

G18_en = 1

# 想要运行哪些phase，如果全部运行，可以写 range(0, 32)
phase = range(0, 2)


delay_L4 = (9*2,  9*2,  9*2,  20*2,  20*2,  20*2)
delay_L5 = (14,  14,  14,  20,  20,  20)

phase_group_id = 0
test_phase = []

#************************************* chip 1, 2 ***************************************#
chip = (1, 2)   # 待修改

if G9_IB_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G9_IB = np.zeros(32)
    phase_en_G9_IB[phase] = 1
    map_config_9_IB = Gen_G9_IB(phase_en=phase_en_G9_IB, clock=15000, out_data_en=G9_en, in_data_en=0, delay_L4=delay_L4, delay_L5=delay_L5)
    for (i, j) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
        map_config[0][phase_group_id][(chip, (i, j))] = map_config_9_IB[0][0][((0, 0), (i, j))]
    for i in range(int(sum(phase_en_G9_IB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G9_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G9 = np.zeros(32)
    phase_en_G9[phase] = 1
    map_config_9 = Gen_G9_Map_Config(phase_en=phase_en_G9, clock=15000, M=8, N=4, out_data_en=G10_en, in_data_en=G9_IB_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(0, 8):
        for j in range(0, 6):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_9[0][0][((0, 0), (i, j))]
    for i in range(int(sum(phase_en_G9))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

    # group_idx_list.append(phase_group_id)
    # for i in range(0, 8):
    #     for j in range(2, 6):
    #         map_config[0][phase_group_id][(chip, (i, j))] = map_config_9[0][0][((0, 0), (i, j))]
    # for i in range(int(sum(phase_en_G9))):
    #     test_phase.append((phase_group_id, i + 1))
    # phase_group_id += 1


if G10_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G10 = np.zeros(32)
    phase_en_G10[phase] = 1
    map_config_10 = Gen_G10_Map_Config(phase_en=phase_en_G10, clock=15000, M=8, N=4, out_data_en=G11_en, in_data_en=G9_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(0, 8):
        for j in range(6, 10):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_10[0][0][((0, 0), (i - 0, j - 6))]
    for i in range(int(sum(phase_en_G10))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G11_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G11 = np.zeros(32)
    phase_en_G11[phase] = 1
    map_config_11 = Gen_G11_Map_Config(phase_en=phase_en_G11, clock=15000, M=8, N=4, out_data_en=G12_en, in_data_en=G10_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(8, 16):
        for j in range(6, 10):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_11[0][0][((0, 0), (i - 8, j - 6))]
    for i in range(int(sum(phase_en_G11))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G12_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G12 = np.zeros(32)
    phase_en_G12[phase] = 1
    map_config_12 = Gen_G12_Map_Config(phase_en=phase_en_G12, clock=15000, M=8, N=4, out_data_en=G12_OB_en, in_data_en=G11_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(8, 16):
        for j in range(2, 6):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_12[0][0][((0, 0), (i - 8, j - 2))]
    for i in range(int(sum(phase_en_G12))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G12_OB_en == 1:
    #
    #
    group_idx_list.append(phase_group_id)
    phase_en_G12_OB = np.zeros(32)
    phase_en_G12_OB[phase] = 1
    map_config_12_OB = Gen_G12_OB(phase_en=phase_en_G12_OB, clock=15000, out_data_en=G13_IB_en, in_data_en=G12_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
        map_config[0][phase_group_id][(chip, (core_x, core_y))] = map_config_12_OB[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G12_OB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

#************#
if G12_OB_en == 1 and G13_IB_en == 1:
    group_idx_list.append(phase_group_id)
    phase_en_G12_G13 = np.zeros(32)
    phase_en_G12_G13[phase] = 1
    map_config_12_G13 = Gen12_Gen13_relay_cores(phase_en=phase_en_G12_G13, clock=15000, out_data_en=1, in_data_en=1, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9)]:
        map_config[0][phase_group_id][((1, 1), (core_x, core_y))] = map_config_12_G13[0][0][((0, 0), (core_x, core_y))]
    phase_group_id += 1

    group_idx_list.append(phase_group_id)
    for (core_x, core_y) in [(5, 0), (7, 0), (9, 0), (11, 0)]:
        map_config[0][phase_group_id][((2, 2), (core_x, core_y))] = map_config_12_G13[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G12_G13))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1
#************#

#************************************* chip 2, 1 ***************************************#
chip = (2, 1)   # 待修改

if G13_IB_en == 1:
    # group 7
    #
    # 4
    group_idx_list.append(phase_group_id)
    phase_en_G13_IB = np.zeros(32)
    phase_en_G13_IB[phase] = 1
    map_config_13_IB = Gen_G13_IB(phase_en=phase_en_G13_IB, clock=15000, out_data_en=G13_en, in_data_en=G12_OB_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9), (11, 9)]:
        map_config[0][phase_group_id][(chip, (core_x, core_y))] = map_config_13_IB[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G13_IB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G13_en == 1:
    # group 6
    # 4
    # 4
    group_idx_list.append(phase_group_id)
    phase_en_G13 = np.zeros(32)
    phase_en_G13[phase] = 1
    map_config_13 = Gen_G13_Map_Config(phase_en=phase_en_G13, clock=15000, M=8, N=4, out_data_en=G14_en, in_data_en=G13_IB_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(8, 16):
        for j in range(5, 9):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_13[0][0][((0, 0), (i - 8, j - 5))]
    for i in range(int(sum(phase_en_G13))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G14_en == 1:
    # group 5
    # 4个phase接收Group13的数据
    # 1个phase发送给Group15
    group_idx_list.append(phase_group_id)
    phase_en_G14 = np.zeros(32)
    phase_en_G14[phase] = 1
    map_config_14 = Gen_G14_Map_Config(phase_en=phase_en_G14, clock=15000, M=8, N=4, out_data_en=G15_en, in_data_en=G13_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(0, 8):
        for j in range(5, 9):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_14[0][0][((0, 0), (i - 0, j - 5))]
    for i in range(int(sum(phase_en_G14))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G15_en == 1:
    # group 4
    # 1个phase接收Group14的数据
    # 2个phase发送给Group15 OB
    group_idx_list.append(phase_group_id)
    phase_en_G15 = np.zeros(32)
    phase_en_G15[phase] = 1
    map_config_15 = Gen_G15_Map_Config(phase_en=phase_en_G15, clock=15000, M=16, N=4, out_data_en=G15_OB_en, in_data_en=G14_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(16):
        for j in range(1, 5):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_15[0][0][((0, 0), (i, j - 1))]
    for i in range(int(sum(phase_en_G15))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G15_OB_en == 1:
    # group 3
    # 2个phase接收Group15的数据
    #
    group_idx_list.append(phase_group_id)
    phase_en_G15_OB = np.zeros(32)
    phase_en_G15_OB[phase] = 1
    map_config_15_OB = Gen_G15_OB(phase_en=phase_en_G15_OB, clock=15000, out_data_en=G16_IB_en, in_data_en=G15_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
        map_config[0][phase_group_id][(chip, (core_x, core_y))] = map_config_15_OB[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G15_OB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

#************************************* chip 2, 0 ***************************************#
chip = (2, 0)

if G16_IB_en == 1:
    # group 2
    #
    # 2个phase发送给Group16

    group_idx_list.append(phase_group_id)
    phase_en_G16_IB = np.zeros(32)
    phase_en_G16_IB[phase] = 1
    map_config_16_IB = Gen_G16_IB(phase_en=phase_en_G16_IB, clock=22000, in_data_en=G15_OB_en, out_data_en=G16_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
        map_config[0][phase_group_id][(chip, (core_x, core_y))] = map_config_16_IB[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G16_IB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G16_en == 1:
    # group 1
    # 2个phase接收Group16 IB的数据
    # 2个phase发送给Group17
    group_idx_list.append(phase_group_id)
    phase_en_G16 = np.zeros(32)
    phase_en_G16[phase] = 1
    map_config_16 = Gen_G16_Map_Config(phase_en=phase_en_G16, clock=22000, M=16, N=4, start_row=0, in_data_en=G16_IB_en, out_data_en=G17_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(16):
        for j in range(5, 9):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_16[0][0][((0, 0), (i, j - 5))]
    for i in range(int(sum(phase_en_G16))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

if G17_en == 1:
    # group 0
    # 2个phase接收Group16的数据
    #
    group_idx_list.append(phase_group_id)
    phase_en_G17 = np.zeros(32)
    phase_en_G17[phase] = 1
    map_config_17 = Gen_G17_Map_Config(phase_en=phase_en_G17, clock=22000, M=16, N=4, start_row=0, in_data_en=G16_en, out_data_en=G18_en, delay_L4=delay_L4, delay_L5=delay_L5)
    for i in range(16):
        for j in range(1, 5):
            map_config[0][phase_group_id][(chip, (i, j))] = map_config_17[0][0][((0, 0), (i, j - 1))]
    for i in range(int(sum(phase_en_G17))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1

#************************************* chip 1, 0 ***************************************#
chip = (1, 0)
if G18_en == 1:

    group_idx_list.append(phase_group_id)
    phase_en_G18_IB = np.zeros(32)
    phase_en_G18_IB[phase] = 1
    map_config_18_IB = Gen_G18_IB(phase_en=phase_en_G18_IB, clock=22000, in_data_en=G17_en, out_data_en=0, delay_L4=delay_L4, delay_L5=delay_L5)
    for (core_x, core_y) in [(15, 0)]:
        map_config[0][phase_group_id][(chip, (core_x, core_y))] = map_config_18_IB[0][0][((0, 0), (core_x, core_y))]
    for i in range(int(sum(phase_en_G18_IB))):
        test_phase.append((phase_group_id, i + 1))
    phase_group_id += 1
#***************************************************************************************#


map_config = add_router_info(map_config=map_config, group_idx_list=group_idx_list, chip_x_num=3, chip_y_num=3, core_x_num=16, core_y_num=10)
map_config = change_ir_to_prims(map_config=map_config, group_idx_list=group_idx_list, chip_x_num=3, chip_y_num=3, core_x_num=16, core_y_num=10)

from generator.test.SCH_Mapping.changeIR import changeIR
map_config = changeIR(map=map_config, chip_x=3, chip_y=3, group_idx_list=group_idx_list)


if self_adopt:
    map_config['sim_clock'] = min(400000, len(phase) * 50000)


if save:
    import pickle
    with open('chip_2_0_LUT.map_config', 'wb') as f:
        pickle.dump(map_config, f)

if run:
    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    test_config = {
            'tb_name': case_file_name,
            'test_mode': TestMode.MEMORY_STATE,
            'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.dict,
            # 'debug_file_switch': HardwareDebugFileSwitch().close_all.debug_top.dict,
            # 'debug_file_switch': HardwareDebugFileSwitch().open_debug_message.dict,
            'test_group_phase': test_phase
        }
    if debug_top:
        test_config['debug_file_switch'] = HardwareDebugFileSwitch().close_all.open_debug_message.multi_chip.dict

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()

if save_LUT:
    from generator.test.get_router_info import get_router_info

    router_static = get_router_info(map_config)

    import scipy.io as scio

    scio.savemat('./chip_2_1_LUT.mat', {'router_static': router_static})