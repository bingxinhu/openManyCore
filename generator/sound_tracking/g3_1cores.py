import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.sound_tracking.gru_data import generate_gru_data
from generator.sound_tracking.utils import get_core_id
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.sound_tracking.quantization_config import QuantizationConfig
from generator.sound_tracking.g3_data import generate_g3_data


def gen_3_map_config(phase_en, clock_in_phase, size_x, size_y, handler, data=None, delay_l4=None, delay_l5=None,
                     in_data_en=False, chip=(0, 0)):
    """
        Sound Tracking: Group 3
        core_x * core_y:  * 1
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    # ******** 数据交互 ********
    offset = 1
    # 接收 256
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f000 >> 2,
                       addr_inb=0x1f000 >> 2, addr_bias=0x1f000 >> 2, addr_out=0x1f000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=1536 // 8 - 1, addr_rhead_base=0x300,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            soma2 = p06(addr_in=0x8380 + ((39 * 32) >> 2), addr_out=0x0000 >> 2,
                        addr_ciso=0x8380 + ((39 * 32 + 128 + 16) >> 2), length_in=128,
                        length_out=256 + 16, length_ciso=128 + 16, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                        data_in=None, data_ciso=None)
            soma2 = None if not in_data_en else soma2
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # Linear + LUT
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            data_x = data['g3_input'][(core_x, core_y)] if not in_data_en else None
            data_w = data['g3_weight'][(core_x, core_y)]
            data_b = data['g3_bias'][(core_x, core_y)]
            data_lut = handler.lut_gen('tanh', 1, 8, handler.qconfig['mlp_tanh_d'], 8,
                                       handler.qconfig['mlp_tanh_m'])
            axon = p04(cin=256 + 1, cout=2, addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x12000 >> 2,
                       addr_out=0x12080 >> 2, ina_type=1, inb_type=1, load_bias=0,
                       data_x=data_x, data_w=data_w, data_b=None)
            axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                       addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                       load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=data_x, data_w=data_w, data_b=None)
            soma1 = None
            router = None
            soma2 = p07(addr_in=0x12080 >> 2, addr_out=0x0100 >> 2, addr_lut=0x12100 >> 2,
                        group_num=1, neuron_num=2,
                        lut_dw=1, type_in=0, type_out=1, in_cut_start=handler.qconfig['mlp'],
                        row_ck_on=0, in_row_max=1, data_in=None, data_lut=data_lut)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'ST_3'
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[phase_offset + 0] = 1

    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    g3_data = generate_g3_data(handler=handler, size_x=1, size_y=1)

    clock_in_phase = 500
    config = gen_3_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, handler=handler,
                              data=g3_data, in_data_en=False, out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

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
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
