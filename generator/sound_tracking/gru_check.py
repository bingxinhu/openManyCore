from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.sound_tracking.gru_data import generate_gru_data
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.sound_tracking.utils import get_core_id
from generator.sound_tracking.quantization_config import QuantizationConfig


def check(ref_data, compare: ResultCompare, sequence_length, size_y, size_x, chip=(0, 0), empty_num=0):
    data = {}
    offset = 1

    data[empty_num + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + 0][chip, (core_x, core_y)] = [{
                'data': ref_data['gru_init']['x'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x0000 >> 2,
                'type': 1
            },
            {
                'data': ref_data['gru_init']['h'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x1fee0 >> 2,
                'type': 1
            }]

    data[empty_num + offset + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 0][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU1_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]
    
    data[empty_num + offset + 1] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 1][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU2_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 2] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 2][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['r_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0880 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 3] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 3][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU4_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 4] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 4][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU5_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 5] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 5][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['zt_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0900 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 6] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 6][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU7_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 7] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 7][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU8_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0980 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 8] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 8][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU9_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 9] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 9][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['n_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0880 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 10] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 10][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU11_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0980 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 11] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 11][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU12_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 12] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 12][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['GRU13_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 13] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 13][chip, (core_x, core_y)] = [{
                'data': ref_data['forward']['next_hid_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xfee0 >> 2,
                'type': 1
            }]

    data[empty_num + offset + 14] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 14][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU1_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 15] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 15][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU2_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 16] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 16][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['r_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0880 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 17] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 17][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU4_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 18] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 18][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU5_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 19] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 19][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['zt_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0900 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 20] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 20][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU7_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 21] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 21][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU8_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0980 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 22] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 22][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU9_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 23] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 23][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['n_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0880 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 24] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 24][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU11_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0980 + 0x260) >> 2,
                'type': 1
            }]

    data[empty_num + offset + 25] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 25][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU12_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0280 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 26] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 26][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['GRU13_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': (0x0480 + 0x260) >> 2,
                'type': 0
            }]

    data[empty_num + offset + 27] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        idx = get_core_id(core_x - size_x[0], core_y - size_y[0])
        if idx < sequence_length:
            data[empty_num + offset + 27][chip, (core_x, core_y)] = [{
                'data': ref_data['backward']['next_hid_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xff70 >> 2,
                'type': 1
            }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_gru_data(handler, size_y=3, size_x=16)

    compare = ResultCompareWithClockSpecificSimulator(
        data_file_name='ST_GRU', save_ref_data_en=True, phase_en=None,
        print_matched=True, step=0)
    check(data, compare, size_y=(0, 3), size_x=(0, 16), chip=(0, 0), 
          sequence_length=handler.sequence_length)
    compare.run()
    compare.show_result()
