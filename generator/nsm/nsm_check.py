from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.nsm.nsm_data import generate_nsm_data
from generator.nsm.nsm_data_handler import NSMDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(ref_data, compare, size_y, size_x, chip=(0, 0), offset=0):
    data = {}
    for i in range(32):
        data[i] = {}
    offset += 1

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[offset + 0][chip, (core_x, core_y)] = [{
            'data': ref_data['t1_cut'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x111a0 >> 2,
            'type': 1
        }]
        data[offset + 1][chip, (core_x, core_y)] = [{
            'data': ref_data['t2_cut'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': (0x111a0 + 128) >> 2,
            'type': 1
        }]
        data[offset + 2][chip, (core_x, core_y)] = [{
            'data': ref_data['hidden_1_cut'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x111a0 >> 2,
            'type': 1
        }]
        data[offset + 3][chip, (core_x, core_y)] = [{
            'data': ref_data['t3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0020 >> 2,
            'type': 0
        }]
        data[offset + 4][chip, (core_x, core_y)] = [{
            'data': ref_data['t4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': (0x0020 + 16 * 4) >> 2,
            'type': 0
        }]
        data[offset + 5][chip, (core_x, core_y)] = [{
            'data': ref_data['hidden_2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x111a0 >> 2,
            'type': 0
        }]
        print(ref_data['hidden_2'][(0, 0)])
        data[offset + 6][chip, (core_x, core_y)] = [{
            'data': ref_data['act_fun'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x111a0 >> 2,
            'type': 1
        }]
        data[offset + 7][chip, (core_x, core_y)] = [{
            'data': ref_data['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0020 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = NSMDataHandler(pretrained=False)
    data = generate_nsm_data(handler, size_y=1, size_x=1)

    compare = ResultCompareWithClockSpecificSimulator(
        data_file_name='nsm_000', save_ref_data_en=True, phase_en=None,
        print_matched=True, step=0)

    check(data, compare, size_y=(0, 1), size_x=(0, 1), chip=(0, 0))
    compare.run()
    compare.show_result()
