from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.sound_tracking_dynamic.g1_data import generate_g1_data
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(ref_data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_num=0):
    data = {}

    data[empty_num + 0] = {}
    data[empty_num + 0][chip, (12, 0)] = [{
        'data': ref_data['g3_output'][(0, 0)],
        'addr_start': 0x8380,
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
