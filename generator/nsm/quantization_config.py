import warnings


class QuantizationConfig:
    def __init__(self):
        self.__quantization_config = {
            'tt': 3,
            'st': 5,
            'tt_st': 4,
        }

    def __getitem__(self, item):
        result = self.__quantization_config[item]
        return result
