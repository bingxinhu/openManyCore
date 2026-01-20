import os
import sys

sys.path.append(os.getcwd())
from generator.sound_tracking_dynamic.g0_4cores import gen_0_map_config
from generator.sound_tracking_dynamic.g1_4cores import gen_1_map_config
from generator.sound_tracking_dynamic.gru_39cores import gen_gru_map_config
from generator.sound_tracking_dynamic.g3_1cores import gen_3_map_config
from generator.mapping_utils.map_config_gen import MapConfigGen
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
from generator.sound_tracking_dynamic.data_gen import sound_tracking_data
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.mapping_utils.prims import p41


def main():

    config = MapConfigGen()
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data_all = sound_tracking_data()

    clock_in_phase = 50_0000
    phase = np.zeros(50).astype(int)
    delay_l4 = (80,) * 9
    delay_l5 = (80,) * 9

    send_to_fpga = True

    # **** Group En ****
    g0_en = 1
    g1_en = 1
    gru_en = 1
    g3_en = 1

    # **** Phase En ****
    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    phase[4] = 0
    phase[5] = 0
    phase[6] = 0
    phase[7] = 0

    phase[:] = 1

    clock_0 = 140_0000 - 1
    clock_1 = 140_0000
    step_exe_number = 1
    config.sim_clock = clock_1 * 1
    init_data = True

    if g0_en:
        g0_config = gen_0_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1, data=data_all,
                                     out_data_en=g1_en, chip=(0, 0), cuts=None, in_data_en=g3_en,
                                     delay_l4=delay_l4, delay_l5=delay_l5, send_to_fpga=send_to_fpga)
        config.add_config(g0_config, core_offset=(12, 0), clock_in_phase=None, phase_adaptive=True)

    if g1_en:
        g1_config = gen_1_map_config(phase_en=phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1, data=data_all,
                                     out_data_en=gru_en, chip=(0, 0), cuts=handler.qconfig['conv1'], in_data_en=g0_en,
                                     init_data=init_data, delay_l4=delay_l4, delay_l5=delay_l5)
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
                                     data=data_all, in_data_en=gru_en, chip=(0, 0), out_data_en=g0_en,
                                     delay_l4=delay_l4, delay_l5=delay_l5, init_data=init_data,
                                     send_to_fpga=send_to_fpga)
        config.add_config(g3_config, core_offset=(7, 3), clock_in_phase=None, phase_adaptive=True)

    MapConfigGen.add_router_info(config.map_config)
    # MapConfigGen.set_step_clock(config.map_config, clock_0=clock_0, clock_1=clock_1)
    # MapConfigGen.set_step_exe_number(config.map_config, step_exe_number)

    prim = {
        'axon': p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                    addr_inb=0x1f000 >> 2, addr_bias=0x1f000 >> 2, addr_out=0x1f000 >> 2, axon_delay=True,
                    L4_num=5, L5_num=5, A2S2_mode=True),
        'soma1': None, 'router': None, 'soma2': None
    }
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    return config.map_config


if __name__ == '__main__':
    main()
