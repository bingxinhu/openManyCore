from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.sound_tracking.g1_data import generate_g1_data
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.sound_tracking.utils import get_core_id
from generator.sound_tracking.quantization_config import QuantizationConfig


def check(ref_data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_num=0):
    data = {}
    offset = 1

    data[empty_num + offset + 1] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[empty_num + offset + 1][chip, (core_x, core_y)] = [{
            'data': ref_data['conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10980 >> 2,
            'type': 1
        }]

    data[empty_num + offset + 2] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[empty_num + offset + 2][chip, (core_x, core_y)] = [{
            'data': ref_data['conv1']['input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]

    # data[empty_num + offset + 3] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data[empty_num + offset + 3][chip, (core_x, core_y)] = [{
    #         'data': ref_data['relu1']['output'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x0000 >> 2,
    #         'type': 1
    #     }]

    data[empty_num + offset + 9] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[empty_num + offset + 9][chip, (core_x, core_y)] = [{
            'data': ref_data['max_pool']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xc800 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_g1_data(handler, size_y=1, size_x=4)

    compare = ResultCompareWithClockSpecificSimulator(
        data_file_name='ST_1', save_ref_data_en=True, phase_en=4,
        print_matched=True, step=0)
    check(data, compare, size_y=(0, 1), size_x=(0, 4), chip=(0, 0))
    compare.run()
    compare.show_result()
