import warnings
from math import log


class QuantizationConfig:
    def __init__(self, name='obstacle'):
        self.__name = name
        self.__quantization_config = {
            'obstacle': {
                'in_cut_start': {
                    'conv1': 3,
                    'res1': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    },
                    'conv2': 5,
                    'conv3': 5,
                    'res2': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    },
                    'res3': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    }
                }
            },
            'mouse': {
                'in_cut_start': {
                    'conv1': 3,
                    'res1': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    },
                    'conv2': 2,
                    'conv3': 5,
                    'res2': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    },
                    'res3': {
                        'conv1': 5,
                        'conv2': 5,
                        'add': 0
                    }
                }
            },
            'sd': {
                'in_cut_start': {
                    'conv1': int(log(16, 4)),
                    'res1': {
                        'conv1': int(log(256, 4)),
                        'conv2': int(log(1024, 4)),
                        'add': int(log(1, 4))
                    },
                    'conv2': int(log(256, 4)),
                    'conv3': int(log(256, 4)),
                    'res2': {
                        'conv1': int(log(1024, 4)),
                        'conv2': int(log(1024, 4)),
                        'add': int(log(1, 4))
                    },
                    'res3': {
                        'conv1': int(log(1024, 4)),
                        'conv2': int(log(1024, 4)),
                        'add': int(log(1, 4))
                    }
                }
            }
        }

    def get_sd_para(self):
        self.__quantization_config = {
            'weight':
                {
                    'conv1': 16,
                    'res1': {
                        'conv1': 256,
                        'conv2': 1024
                    },
                    'conv2': 256,
                    'conv3': 256,
                    'res2': {
                        'conv1': 1024,
                        'conv2': 1024
                    },
                    'res3': {
                        'conv1': 1024,
                        'conv2': 1024
                    },
                    'conv4': 1024
                }
        }

    def __getitem__(self, item):
        result = self.__quantization_config.get(self.__name).get(item)
        if result is None:
            warnings.warn('\'{}\' is not a quantization parameter!'.format(item))
        return result
