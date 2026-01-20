import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.nsm.nsm_data import generate_nsm_data


def gen_nsm_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                       in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False,
                       delay_l4=None, delay_l5=None):
    """
        NSM
        core_x * core_y: 1 * 1
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
    # 接收16B*2；
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x6E40 >> 2 if core_x == size_x - 1 else 0x7E00 >> 2,
                            addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2,
                            length_in=63 * 64, num_in=3,
                            length_ciso=1, num_ciso=3,
                            length_out=63 * 64, num_out=3,
                            type_in=1, type_out=1,
                            data_in=data['res1']['add']['output1'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en,
                         send_num=3 * 63 * 64 // 8 - 1, receive_num=0,
                         addr_din_base=0x400, addr_din_length=16 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=3 * 63 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2,
                            length_in=32, num_in=1, length_ciso=1, num_ciso=1, length_out=32, num_out=1,
                            type_in=1, type_out=1, data_in=None, data_ciso=None)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # Linear TT
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=4, cout=100, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10280 >> 2, addr_bias=0x0, addr_out=0x0020 >> 2,
                       ina_type=1, inb_type=1, load_bias=0,
                       data_x=None if in_data_en else data['inputs'][(core_x, core_y)],
                       data_w=data['linear_tt']['weight'][(core_x, core_y)],
                       data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x0020 >> 2, addr_out=0x111a0 >> 2,
                        cin=128, cout=128, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['tt'],
                        row_ck_on=0, in_row_max=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Linear ST
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=5, cout=100, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0010 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x0, addr_out=0x0020 >> 2,
                       ina_type=1, inb_type=1, load_bias=0,
                       data_x=None if in_data_en else data['init_state'][(core_x, core_y)],
                       data_w=data['linear_st']['weight'][(core_x, core_y)],
                       data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x0020 >> 2, addr_out=(0x111a0 + 128) >> 2,
                        cin=128, cout=128, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['st'],
                        row_ck_on=0, in_row_max=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # t1_cut * t2_cut
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p03(tensor_en=False, x_array_num=1, cin=128, tensor_px=1, tensor_py=1, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x111a0 >> 2, addr_inb=(0x111a0 + 128) >> 2, addr_bias=0x0000 >> 2,
                       addr_out=0x0020 >> 2, ina_type=1, load_bias=0, bias_length=0,
                       data_x=None, data_y=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x0020 >> 2, addr_out=0x111a0 >> 2,
                        cin=128, cout=112, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['tt_st'],
                        row_ck_on=0, in_row_max=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Linear TS
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=100, cout=5, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x111a0 >> 2, addr_inb=0x10480 >> 2, addr_bias=0x0, addr_out=0x0020 >> 2,
                       ina_type=1, inb_type=1, load_bias=0,
                       data_x=None,
                       data_w=data['linear_ts']['weight'][(core_x, core_y)],
                       data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Linear SS
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=5, cout=5, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0010 >> 2, addr_inb=0x11100 >> 2, addr_bias=0x0, addr_out=(0x0020 + 16 * 4) >> 2,
                       ina_type=1, inb_type=1, load_bias=0,
                       data_x=None,
                       data_w=data['linear_ss']['weight'][(core_x, core_y)],
                       data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # add
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=16, px=1, py=1,
                       addr_in=0x0020 >> 2, addr_bias=0x0, addr_out=0x111a0 >> 2, pad_top=0,
                       pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0x0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # add
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=1, sx=1, sy=1, cin=16, px=1, py=1,
                       addr_in=0x111a0 >> 2, addr_bias=0x0, addr_out=0x0020 >> 2, pad_top=0,
                       pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0x3ffffffe)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x0020 >> 2, addr_out=0x111a0 >> 2,
                        cin=16, cout=16, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=15,
                        row_ck_on=0, in_row_max=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x0010 >> 2, addr_out=0x0020 >> 2, addr_ciso=0x111a0 >> 2, length_in=16,
                        num_in=1, length_ciso=16, num_ciso=1, length_out=32, num_out=1, type_in=1,
                        type_out=1, data_in=None, data_ciso=None)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.nsm.nsm_data_handler import NSMDataHandler
    from generator.nsm.quantization_config import QuantizationConfig

    case_file_name = 'NSM_000'
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)

    phase[phase_offset + 0] = 1
    phase[phase_offset + 1] = 1
    phase[phase_offset + 2] = 1
    phase[phase_offset + 3] = 0
    phase[phase_offset + 4] = 0

    phase[:] = 1

    handler = NSMDataHandler(pretrained=False, quantization_en=True)
    data = generate_nsm_data(handler, size_y=1, size_x=1)
    qconfig = QuantizationConfig()
    in_cut_start_dict = qconfig

    clock_in_phase = 1_000
    config = gen_nsm_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, data=data,
                                in_data_en=False, out_data_en=False, chip=chip, in_cut_start_dict=in_cut_start_dict)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

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
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
