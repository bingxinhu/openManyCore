from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.detection.ObstacleNet.g2_data import generate_g2_data
from generator.detection.detection_data_handler import DetectionDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(ref_data, compare, size_y, size_x, chip=(0, 0), offset=0):
    data = {}
    offset += 3

    # received data check
    # data[0] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data[0][chip, (core_x, core_y)] = [{
    #         'data': ref_data['res1']['conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x0000 >> 2,
    #         'type': 1
    #     }]
    #
    # data[1] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data[1][chip, (core_x, core_y)] = [{
    #         'data': ref_data['res1']['conv1']['input2'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x2F40 >> 2,
    #         'type': 1
    #     }]
    #
    # data[2] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data[2][chip, (core_x, core_y)] = [{
    #         'data': ref_data['res1']['conv1']['input3'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x5E80 >> 2,
    #         'type': 1
    #     }]

    data[offset + 3] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[offset + 3][chip, (core_x, core_y)] = [{
            'data': ref_data['res1']['conv2']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6E40 >> 2 if core_x == size_x[1] - 1 else 0x7E00 >> 2,
            'type': 1
        }]

    data[offset + 4] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[offset + 4][chip, (core_x, core_y)] = [{
            'data': ref_data['res1']['add']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6E40 >> 2 if core_x == size_x[1] - 1 else 0x7E00 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = DetectionDataHandler(name='obstacle', pretrained=False)
    data = generate_g2_data(handler, size_y=1, size_x=8)

    # compare = ResultCompareWithClockSpecificSimulator(
    #     data_file_name='ST_1', save_ref_data_en=True, phase_en=4,
    #     print_matched=True, step=0)
    compare = ResultCompare(data_2_file_name='Obstacle_2', save_data_1_en=True)
    check(data, compare, size_y=(0, 1), size_x=(0, 8), chip=(0, 0))
    compare.run()
    compare.show_result()
