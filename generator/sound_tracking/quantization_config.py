import warnings


class QuantizationConfig:
    def __init__(self, sequence_length):
        self.__quantization_config = {
            'in_cut_start': [{'forward':
                                  {'GRU3_cut': 4,
                                   'GRU6_cut': 4,
                                   'GRU8_cut': 4,
                                   'GRU10_cut': 4,
                                   'GRU11_cut': 0,
                                   'next_hid_cut': 4},
                              'backward':
                                  {'GRU3_cut': 4,
                                   'GRU6_cut': 4,
                                   'GRU8_cut': 4,
                                   'GRU10_cut': 4,
                                   'GRU11_cut': 0,
                                   'next_hid_cut': 4}}
                             ] * sequence_length,
            'q_one': [{'forward': 127 * 2, 'backward': 127 * 2}] * sequence_length,
            'lut': [{'forward':
                         {'sigmoid_r_d': 128,
                          'sigmoid_r_m': 128 * 2,
                          'sigmoid_z_d': 128,
                          'sigmoid_z_m': 128 * 2,
                          'tanh_d': 128,
                          'tanh_m': 128},
                     'backward':
                         {'sigmoid_r_d': 128,
                          'sigmoid_r_m': 128 * 2,
                          'sigmoid_z_d': 128,
                          'sigmoid_z_m': 128 * 2,
                          'tanh_d': 128,
                          'tanh_m': 128}}
                    ] * sequence_length,
            'conv1': 4,
            'mlp': 4,
            'mlp_tanh_d': 128,
            'mlp_tanh_m': 128
        }

    def __getitem__(self, item):
        result = self.__quantization_config.get(item)
        if result is None:
            warnings.warn('\'{}\' is not a quantization parameter!'.format(item))
        return result
