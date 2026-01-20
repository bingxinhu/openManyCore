import os
import sys

sys.path.append(os.getcwd())
from generator.sound_tracking.g1_4cores import gen_1_map_config
from generator.sound_tracking.gru_39cores import gen_gru_map_config
from generator.sound_tracking.g3_1cores import gen_3_map_config
from generator.mapping_utils.map_config_gen import MapConfigGen
import numpy as np
from generator.mapping_utils.prims import p41, p06
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
from generator.sound_tracking.data_gen import sound_tracking_data
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler


def main():
    case_file_name = 'ST_052'

    config = MapConfigGen()
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data_all = sound_tracking_data()

    clock_in_phase = 200_000
    phase = np.zeros(50).astype(int)
    delay_l4 = (28,) * 9
    delay_l5 = (28,) * 9

    # **** Group En ****
    g1_en = 1
    gru_en = 1
    g3_en = 1

    # **** Phase En ****
    phase[0] = 0
    phase[1] = 0
    phase[2] = 0
    phase[3] = 0
    phase[4] = 0
    phase[5] = 0
    phase[6] = 0
    phase[7] = 0

    phase[:] = 1

    clock_0 = 350_000 - 1
    clock_1 = 350_000
    step_exe_number = 1
    config.sim_clock = clock_1 * 43
    init_data = False

    if g1_en:
        g1_config = gen_1_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1, data=data_all,
                                     out_data_en=gru_en, chip=(0, 0), cuts=handler.qconfig['conv1'],
                                     init_data=init_data)
        config.add_config(g1_config, core_offset=(0, 0), clock_in_phase=None, phase_adaptive=True)

    if gru_en:
        gru_config = gen_gru_map_config(phase, clock_in_phase=clock_in_phase, size_x=16, size_y=3, data=data_all,
                                        in_data_en=g1_en, out_data_en=g3_en, chip=(0, 0),
                                        in_cut_start_mat=handler.qconfig['in_cut_start'],
                                        q_ones=handler.qconfig['q_one'], lut_mat=handler.qconfig['lut'],
                                        handler=handler, init_data=init_data,
                                        delay_l4=delay_l4, delay_l5=delay_l5)
        config.add_config(gru_config, core_offset=(0, 1), clock_in_phase=None, phase_adaptive=True)

    if g3_en:
        g3_config = gen_3_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, handler=handler,
                                     data=data_all, in_data_en=gru_en, chip=(0, 0),
                                     delay_l4=delay_l4, delay_l5=delay_l5)
        config.add_config(g3_config, core_offset=(7, 3), clock_in_phase=None, phase_adaptive=True)

    MapConfigGen.add_router_info(config.map_config)
    MapConfigGen.set_step_clock(config.map_config, clock_0=clock_0, clock_1=clock_1)
    MapConfigGen.set_step_exe_number(config.map_config, step_exe_number)

    prim = {
        'axon': p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                    addr_inb=0x1f000 >> 2, addr_bias=0x1f000 >> 2, addr_out=0x1f000 >> 2, axon_delay=True,
                    L4_num=20, L5_num=20, A2S2_mode=True),
        'soma1': None, 'router': None, 'soma2': None
    }
    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0, length_in=256, length_out=256,
    #                  length_ciso=1, num_in=40, num_ciso=40, num_out=40, type_in=1, type_out=1,
    #                  in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
    # }

    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
    del_command = 'rd/s/q cmp_out'
    os.system(del_command)
    os.chdir(c_path)

    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.close_burst.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test()


if __name__ == '__main__':
    main()
