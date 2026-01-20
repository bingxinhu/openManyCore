from generator.nsm.nsm_1core import gen_nsm_map_config
import os
from generator.nsm.nsm_data_handler import NSMDataHandler
from generator.nsm.quantization_config import QuantizationConfig
from generator.nsm.nsm_data import generate_nsm_data
import numpy as np
from generator.mapping_utils.map_config_gen import MapConfigGen


def main():
    handler = NSMDataHandler(pretrained=False, quantization_en=True)
    data = generate_nsm_data(handler, size_y=1, size_x=1)
    qconfig = QuantizationConfig()
    in_cut_start_dict = qconfig

    phase = np.zeros(50).astype(int)

    phase[:] = 1

    clock_in_phase = 1_0000

    map_config = MapConfigGen()

    # Bias of ST Net
    empty_config = {
        'sim_clock': None,
        ((0, 0), 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for i in range(17):
        map_config.add_config(empty_config, core_offset=(0, 0), clock_in_phase=None, phase_adaptive=True)

    config = gen_nsm_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, data=data,
                                in_data_en=False, out_data_en=False, chip=(0, 0), in_cut_start_dict=in_cut_start_dict)
    MapConfigGen.add_router_info(map_config=config)

    map_config.add_config(config, core_offset=(0, 9))

    # delete empty map_config
    for i in range(17):
        map_config.map_config[((0, 0), 0)].pop(i)

    map_config.map_config['sim_clock'] = 200_000

    return map_config.map_config
