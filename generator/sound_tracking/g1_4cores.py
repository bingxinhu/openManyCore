import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.sound_tracking.utils import get_core_id


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, cuts: None, data=None,
                     out_data_en=False, chip=(0, 0), init_data=None):
    """
        Sound Tracking: Group 1
        core_x * core_y: 4 * 1
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
    # 发送 9， 10， 10， 10 * 16
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            length_in = 9 * 32 if core_x == 0 else 10 * 32
            if out_data_en:
                soma1 = p06(addr_in=0xd000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=length_in,
                            length_out=length_in, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['max_pool']['output_with_pad'][(core_x, core_y)] if init_data else None,
                            data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=0, send_num=length_in // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                ad = [0, 9 * 4, 19 * 4, 29 * 4]
                router.addRHead(S=0, T=1, P=0, Q=0, X=0 - core_x, Y=1 - core_y, A=ad[core_x],
                                pack_per_Rhead=length_in // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # ******** 开始计算 ********

    # 放置数据
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_x in [3]:
                length = 11
            else:
                length = 10
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=16 * length,
                        length_out=1, length_ciso=1, num_in=257, num_ciso=257, num_out=257, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                        data_in=data['conv1']['input'][(core_x, core_y)],
                        data_ciso=None)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送右侧交叠数据
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=(0x0000 + 9 * 16) >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=16 * 10,
                        length_out=16, length_ciso=1, num_in=257, num_ciso=257, num_out=257, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                        data_in=None,
                        data_ciso=None)
            if core_x in [3]:
                soma1 = None
                send_en, receive_en = 0, 1
            elif core_x in [0]:
                send_en, receive_en = 1, 0
            else:
                send_en, receive_en = 1, 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=send_en,
                         receive_en=receive_en, send_num=257 * 16 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=257 * 16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if send_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=257 * 16 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10980 >> 2, addr_ciso=0x0000 >> 2, length_in=16,
                        length_out=16 * 12 if core_x == 3 else 16 * 11, length_ciso=16 * 11 if core_x == 3 else 16 * 10,
                        num_in=257, num_ciso=257, num_out=257, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not receive_en:
                soma2 = p06(addr_in=0x0000 >> 2, addr_out=0x10980 >> 2, addr_ciso=0x0000 >> 2, length_in=16 * 10,
                            length_out=16 * 10, length_ciso=1,
                            num_in=257, num_ciso=257, num_out=257, type_in=1, type_out=1,
                            in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送左侧交叠数据
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                length = 12
                send_en, receive_en = 1, 0
            elif core_x in [0]:
                length = 10
                send_en, receive_en = 0, 1
            else:
                length = 11
                send_en, receive_en = 1, 1
            axon = None
            if send_en:
                soma1 = p06(addr_in=(0x10980 + 16) >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=16 * length,
                            length_out=16, length_ciso=1, num_in=257, num_ciso=257, num_out=257, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=None, data_ciso=None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=send_en,
                         receive_en=receive_en, send_num=257 * 16 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=257 * 16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if send_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=257 * 16 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if receive_en:
                soma2 = p06(addr_in=0x10980 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x8380, length_in=16 * length,
                            length_out=16 * (length + 1), length_ciso=16,
                            num_in=257, num_ciso=257, num_out=257, type_in=1, type_out=1,
                            in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            else:
                soma2 = p06(addr_in=0x10980 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x8380, length_in=16 * length,
                            length_out=16 * length, length_ciso=1,
                            num_in=257, num_ciso=257, num_out=257, type_in=1, type_out=1,
                            in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 1/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 0 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=data['conv1']['weight'][(core_x, core_y)],
                       data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 0 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 2/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 42 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 6 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 3/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 84 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 12 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 4/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 126 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 18 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 5/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 168 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 24 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv + MaxPool 1/3 - 6/6
    # [(0, 45), (42, 87), (84, 129), (126, 171), (168, 213), (210, 255)]
    # [(0, 43), (42, 85), (84, 127), (126, 169), (168, 211), (210, 253)]
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            elif core_x in [0]:
                px, pad_left, pad_right = 11, 0, 0
                px_out = 9
            else:
                px, pad_left, pad_right = 12, 0, 0
                px_out = 10
            axon = p41(px=px, py=45, cin=8, cout=16, kx=3, ky=3, sx=1, sy=1, addr_ina=(0x0000 + 210 * px * 16) >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x10900 >> 2, addr_out=0x10980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=pad_left, pad_right=pad_right,
                       data_x=None, data_w=None, data_b=None)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=(0xc800 + 30 * px_out * 16) >> 2,
                        cin=32, cout=16, px=px_out, py=43,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # MaxPool 3/3
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [3]:
                px_out = 10
            elif core_x in [0]:
                px_out = 9
            else:
                px_out = 10
            axon = None
            soma1 = pX5(mode='max', addr_in=0xc800 >> 2, addr_out=0x10980 >> 2, cin=16, cout=16, px=px_out, py=36,
                        kx=1, ky=8, sx=1, sy=7, cmp_c=0x0, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=0)
            router = None
            soma2 = pX5(mode='max', addr_in=0x10980 >> 2, addr_out=0xc800 >> 2, cin=16, cout=16, px=px_out, py=5,
                        kx=1, ky=5, sx=1, sy=1, cmp_c=0x0, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # pad 127 for output data
    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x in [0]:
                px_out = 9
            else:
                px_out = 10
            axon = None
            soma1 = p06(addr_in=0xc800 >> 2, addr_out=0xd000 >> 2, addr_ciso=0xe000 >> 2,
                        length_in=16, num_in=px_out, length_ciso=16, num_ciso=px_out, length_out=32, num_out=px_out,
                        type_in=1, type_out=1, data_ciso=data['pad_127_for_input_g1'][(core_x, core_y)])
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
    from generator.sound_tracking.g1_data import generate_g1_data
    from generator.sound_tracking.quantization_config import QuantizationConfig

    case_file_name = 'ST_1'
    chip = (0, 0)
    cuts = None
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    # phase[0] = 1
    # phase[1] = 1

    phase[phase_offset + 0] = 1
    phase[phase_offset + 1] = 1
    phase[phase_offset + 2] = 1
    phase[phase_offset + 3] = 1
    phase[phase_offset + 4] = 1

    # phase[14:] = 0

    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_g1_data(handler, size_y=1, size_x=4)
    para = QuantizationConfig(sequence_length=39)

    clock_in_phase = 150_000
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=4, size_y=1, data=data,
                              in_data_en=False, out_data_en=False, chip=chip, cuts=para['conv1'])
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
