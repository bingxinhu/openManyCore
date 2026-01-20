from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())


def check(ref_data_obstacle, ref_data_mouse, compare, size_y, size_x, chip=(0, 0), offset=0):
    data = {}
    offset += 0

    data[offset + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_x - size_x[0] == 0:
            if ref_data_mouse and ref_data_obstacle:
                data[offset + 0][chip, (core_x, core_y)] = [{
                    'data': (ref_data_obstacle['avgpool']['output'][(core_x - size_x[0], core_y - size_y[0])] +
                             ref_data_mouse['avgpool']['output'][(core_x - size_x[0], core_y - size_y[0])]),
                    'addr_start': 0x8380,
                    'type': 0
                }]
            elif ref_data_obstacle:
                data[offset + 0][chip, (core_x, core_y)] = [{
                    'data': ref_data_obstacle['avgpool']['output'][(core_x - size_x[0], core_y - size_y[0])],
                    'addr_start': 0x8380,
                    'type': 0
                }]
            else:
                raise ValueError

    compare.add_ref_data(data)
