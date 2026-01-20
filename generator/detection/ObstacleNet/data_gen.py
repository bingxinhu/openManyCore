from generator.detection.detection_data_handler import DetectionDataHandler
from generator.detection.ObstacleNet.g1_data import generate_g1_data
from generator.detection.ObstacleNet.g2_data import generate_g2_data
from generator.detection.ObstacleNet.g3_data import generate_g3_data
from generator.detection.ObstacleNet.g4_data import generate_g4_data
from generator.detection.ObstacleNet.g5_data import generate_g5_data
from generator.detection.ObstacleNet.g6_data import generate_g6_data


def merge_dict(a: dict, b: dict):
    """
    merge B to A
    """
    for key, value in b.items():
        if a.get(key) is not None:
            raise KeyError('key " {} " is already in dictionary!'.format(key))
        a[key] = value


def detection_data(handler):
    g1_data = generate_g1_data(handler, size_y=2, size_x=8)
    g2_data = generate_g2_data(handler, size_y=1, size_x=8)
    g3_data = generate_g3_data(handler, size_y=1, size_x=8)
    g4_data = generate_g4_data(handler, size_y=1, size_x=4)
    g5_data = generate_g5_data(handler, size_y=1, size_x=2)
    g6_data = generate_g6_data(handler, size_y=1, size_x=1)

    data_all = {}
    merge_dict(data_all, g1_data)
    merge_dict(data_all, g2_data)
    merge_dict(data_all, g3_data)
    merge_dict(data_all, g4_data)
    merge_dict(data_all, g5_data)
    merge_dict(data_all, g6_data)

    return data_all
