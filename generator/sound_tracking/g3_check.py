from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.sound_tracking.g3_data import generate_g3_data
from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(ref_data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_num=0):
    data = {}
    offset = 1

    data[empty_num + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[empty_num + 0][chip, (core_x, core_y)] = [{
            'data': ref_data['g3_input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]

    data[empty_num + offset + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[empty_num + offset + 0][chip, (core_x, core_y)] = [{
            'data': ref_data['g3_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0100 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    data = generate_g3_data(handler, size_y=1, size_x=1)

    compare = ResultCompareWithClockSpecificSimulator(
        data_file_name='ST_3', save_ref_data_en=True, phase_en=None,
        print_matched=True, step=0)
    check(data, compare, size_y=(0, 1), size_x=(0, 1), chip=(0, 0))
    compare.run()
    compare.show_result()
