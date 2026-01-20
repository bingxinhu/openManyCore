import copy
import os
import sys

sys.path.append(os.getcwd())
from generator.detection.ObstacleNet.g0_8cores import gen_0_map_config
from generator.detection.ObstacleNet.g1_16cores import gen_1_map_config
from generator.detection.ObstacleNet.g2_8cores import gen_2_map_config
from generator.detection.ObstacleNet.g3_8cores import gen_3_map_config
from generator.detection.ObstacleNet.g4_4cores import gen_4_map_config
from generator.detection.ObstacleNet.g5_2cores import gen_5_map_config
from generator.detection.ObstacleNet.g6_1cores import gen_6_map_config
from generator.mapping_utils.map_config_gen import MapConfigGen
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
from generator.detection.ObstacleNet.data_gen import detection_data
from generator.detection.detection_data_handler import DetectionDataHandler
from generator.mapping_utils.prims import p41


def main():
    case_file_name = 'Det_088'

    config = MapConfigGen()
    obstacle_handler = DetectionDataHandler(name='obstacle', pretrained=True)
    obstacle_data_all = detection_data(handler=obstacle_handler)
    mouse_handler = DetectionDataHandler(name='mouse', pretrained=True)
    mouse_data_all = detection_data(handler=mouse_handler)

    clock_in_phase = 200_000
    phase = np.zeros(50).astype(int)
    delay_l4 = (28,) * 9
    delay_l5 = (28,) * 9
    send_to_fpga = False
    axon_delay_empty_phase = 0

    # **** Net En ****
    obstacle_en = 1
    mouse_en = 1

    # **** Group En ****
    g0_en = 1
    g1_en = 1  # 1 2
    g2_en = 1  # 1 2ss
    g3_en = 1
    g4_en = 1  # 1 2
    g5_en = 1
    g6_en = 1

    # **** Phase En ****
    phase[0] = 1
    phase[1] = 1
    phase[2] = 0
    phase[3] = 0
    phase[4] = 0
    phase[5] = 0
    phase[6] = 0
    phase[7] = 0
    phase[8] = 0
    phase[9] = 0
    phase[10] = 0
    phase[11] = 0
    phase[12] = 0
    phase[13] = 0
    phase[14] = 0
    phase[15] = 0

    phase[:] = 1

    clock_0 = 350_000 - 1
    clock_1 = 350_000
    step_exe_number = 1
    config.sim_clock = clock_1 * 1
    init_data = True

    # Bias of ST Net
    empty_config = {
        'sim_clock': None,
        ((0, 0), 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for i in range(4):
        config.add_config(empty_config, core_offset=(0, 0), clock_in_phase=None, phase_adaptive=True)

    if obstacle_en:
        if g0_en:
            g0_config = gen_0_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1,
                                         data=obstacle_data_all, out_data_en=g1_en, chip=(0, 0), delay_l4=delay_l4,
                                         delay_l5=delay_l5, mouse_en=mouse_en, obstacle_en=obstacle_en,
                                         send_to_fpga=send_to_fpga, g6_en=g6_en,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g0_config, core_offset=(8, 3), clock_in_phase=None, phase_adaptive=True)

        if g1_en:
            g1_config = gen_1_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=8, size_y=2,
                                         data=obstacle_data_all, out_data_en=g2_en, chip=(0, 0),
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         in_data_en=g0_en, init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         name='obstacle', mouse_net_en=mouse_en,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g1_config, core_offset=(8, 4), clock_in_phase=None, phase_adaptive=True)

        if g2_en:
            g2_config = gen_2_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1,
                                         data=obstacle_data_all, in_data_en=g1_en, out_data_en=g3_en, chip=(0, 0),
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g2_config, core_offset=(8, 6), clock_in_phase=None, phase_adaptive=True)

        if g3_en:
            g3_config = gen_3_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1,
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         data=obstacle_data_all, in_data_en=g2_en, out_data_en=g4_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g3_config, core_offset=(8, 7), clock_in_phase=None, phase_adaptive=True)

        if g4_en:
            g4_config = gen_4_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1,
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         data=obstacle_data_all, in_data_en=g3_en, out_data_en=g5_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g4_config, core_offset=(12, 8), clock_in_phase=None, phase_adaptive=True)

        if g5_en:
            g5_config = gen_5_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=2, size_y=1,
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         data=obstacle_data_all, in_data_en=g4_en, out_data_en=g6_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g5_config, core_offset=(10, 8), clock_in_phase=None, phase_adaptive=True)

        if g6_en:
            g6_config = gen_6_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1,
                                         in_cut_start_dict=obstacle_handler.qconfig['in_cut_start'],
                                         data=obstacle_data_all, in_data_en=g5_en, out_data_en=g0_en or send_to_fpga,
                                         chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5, name='obstacle',
                                         send_to_fpga=send_to_fpga,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g6_config, core_offset=(9, 8), clock_in_phase=None, phase_adaptive=True)

    if mouse_en:
        if g1_en:
            g1_config = gen_1_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=8, size_y=2,
                                         data=mouse_data_all, out_data_en=g2_en, chip=(0, 0),
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         in_data_en=g0_en, init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         name='mouse', mouse_net_en=mouse_en,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g1_config, core_offset=(0, 4), clock_in_phase=None, phase_adaptive=True)

        if g2_en:
            g2_config = gen_2_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1,
                                         data=mouse_data_all, in_data_en=g1_en, out_data_en=g3_en, chip=(0, 0),
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g2_config, core_offset=(0, 6), clock_in_phase=None, phase_adaptive=True)

        if g3_en:
            g3_config = gen_3_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=8, size_y=1,
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         data=mouse_data_all, in_data_en=g2_en, out_data_en=g4_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g3_config, core_offset=(0, 7), clock_in_phase=None, phase_adaptive=True)

        if g4_en:
            g4_config = gen_4_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1,
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         data=mouse_data_all, in_data_en=g3_en, out_data_en=g5_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g4_config, core_offset=(4, 8), clock_in_phase=None, phase_adaptive=True)

        if g5_en:
            g5_config = gen_5_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=2, size_y=1,
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         data=mouse_data_all, in_data_en=g4_en, out_data_en=g6_en, chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g5_config, core_offset=(2, 8), clock_in_phase=None, phase_adaptive=True)

        if g6_en:
            g6_config = gen_6_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1,
                                         in_cut_start_dict=mouse_handler.qconfig['in_cut_start'],
                                         data=mouse_data_all, in_data_en=g5_en, out_data_en=g0_en or send_to_fpga,
                                         chip=(0, 0),
                                         init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5, name='mouse',
                                         send_to_fpga=send_to_fpga,
                                         axon_delay_empty_phase=axon_delay_empty_phase)
            config.add_config(g6_config, core_offset=(1, 8), clock_in_phase=None, phase_adaptive=True)

    # add empty prim at the beginning
    prim = {
        'axon': p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                    addr_inb=(0x1f000 + 32) >> 2, addr_bias=(0x1f000 + 64) >> 2, addr_out=(0x1f000 + 96) >> 2,
                    axon_delay=True,
                    L4_num=5, L5_num=5, A2S2_mode=True),
        'soma1': None, 'router': None, 'soma2': None
    }
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    MapConfigGen.add_router_info(config.map_config)

    # delete empty map_config
    for i in range(4):
        config.map_config[((0, 0), 0)].pop(i)

    # config.map_config[((0, 0), 1)] = config.map_config.pop(((0, 0), 0))

    # MapConfigGen.set_step_clock(config.map_config, clock_0=clock_0, clock_1=clock_1)
    # MapConfigGen.set_step_exe_number(config.map_config, step_exe_number)

    config.map_config['step_clock'] = {}
    config.map_config['step_clock'][((0, 0), 0)] = (clock_0, clock_1)
    config.map_config['step_clock'][((0, 0), 1)] = (clock_0, clock_1)
    config.map_config['step_clock'][((0, 0), 2)] = (clock_0, clock_1)
    config.map_config['step_clock'][((0, 0), 3)] = (clock_0, clock_1)
    # config.map_config['chip_register'] = {}
    # config.map_config['chip_register'][(0, 0)] = {
    #     'PCK_OUT_SEL': [1, 1, 1, 1],
    #     'GFINISH_MODE': [1, 0, 0, 0]
    # }

    # chip_group = config.map_config[((0, 0), 1)]
    # config.map_config[((0, 0), 2)] = {}
    # config.map_config[((0, 0), 2)][11] = chip_group.pop(11)
    # config.map_config[((0, 0), 2)][12] = chip_group.pop(12)
    # config.map_config[((0, 0), 2)][13] = chip_group.pop(13)
    # config.map_config[((0, 0), 2)][14] = chip_group.pop(14)
    # config.map_config[((0, 0), 2)][15] = chip_group.pop(15)
    # config.map_config[((0, 0), 2)][16] = chip_group.pop(16)

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
    del_command = 'rd/s/q cmp_out'
    os.system(del_command)
    os.chdir(c_path)

    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.singla_chip.close_burst.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test()


if __name__ == '__main__':
    main()
