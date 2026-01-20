import numpy as np
import sys
import os

from numpy import core
from numpy.lib.arraypad import pad

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81, p43
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def get_core_id(core_x, core_y):
    return core_x + core_y * 16


def gen_5_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False,
                     delay_l4=None, delay_l5=None):
    """
        MobileNet: Group 5
        core_x * core_y: 14 * 1
    """
    map_config = {
        'sim_clock': None,
        'step_clock': {
            ((0, 0), 0): (70000, 200000)
        },
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
    offset = 0
    # ******** 开始计算 ********
    # 深度可分离卷积
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10480 >> 2,
                       addr_out=0x0000 >> 2, ina_type=1, inb_type=1, load_bias=2, bias_length=128, inb_length=128,
                       data_x=[], data_a=[], data_b=[])
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 2) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 3) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 4) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 5) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 6) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 7) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2}) 
                                                        
    if phase_en[offset + 15]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p43(tensor_en=True, x_array_num=0, tensor_px=28, tensor_py=2, tensor_sx=1, tensor_sy=1,
                       addr_ina=0x18880 >> 2, addr_inb=(0x10000 + 0x100 * 8) >> 2, addr_bias=0x0,
                       addr_out=0x7000 >> 2, ina_type=1, inb_type=1, load_bias=0, bias_length=128, inb_length=128,
                       data_x=None, data_a=[], data_b=None)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 16]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, 
                       px=28, py=2, addr_in=0x0000 >> 2, addr_bias=0x0, addr_out=0x0000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x0000 >> 2, addr_out=0xE000 >> 2,
                        cin=128, cout=128, px=28, py=2,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, 
                        in_cut_start=12, row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 17]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = pX5(mode='max', addr_in=0xE000 >> 2, addr_out=0x24000 >> 2,
                        cin=128, cout=128, px=28, py=2,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, 
                        in_cut_start=0, row_ck_on=0, in_row_max=1)                                                                   
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=True, receive_en=True, 
                         send_num=2 * 28 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=2 * 28 * 128 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1 if core_x % 2 == 0 else -1, Y=0, A=0, pack_per_Rhead=2 * 28 * 128 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0xE000 >> 2 if core_x % 2 == 0 else 0x21000 >> 2, 
                        addr_out=0x0000 >> 2, addr_ciso=0x21000 >> 2 if core_x % 2 == 0 else 0xE000 >> 2, 
                        length_in=128, num_in=2 * 28, length_ciso=128, num_ciso=2 * 28, length_out=256, num_out=2 * 28, 
                        type_in=1, type_out=1, data_in=None, data_ciso=None) 
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 18]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=28, py=2, cin=256, cout=128, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10680 >> 2, addr_bias=0x18680 >> 2, addr_out=0x7000 >> 2, 
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None, data_w=[], data_b=[])
            soma1 = pX5(mode='max', addr_in=0x7000 >> 2, addr_out=0xE000 >> 2,
                        cin=128, cout=128, px=28, py=2,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, 
                        in_cut_start=12, row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'MobileNet5'
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[:] = 1

    clock_in_phase = 50_000
    config = gen_5_map_config(phase, clock_in_phase=clock_in_phase, size_x=14, size_y=1,
                              in_data_en=False, out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)
    config['sim_clock'] = 70000

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
