from generator.sound_tracking.sound_tracking_data_handler import SoundTrackingDataHandler
from generator.sound_tracking.g1_data import generate_g1_data
from generator.sound_tracking.gru_data import generate_gru_data
from generator.sound_tracking.g3_data import generate_g3_data


def merge_dict(a: dict, b: dict):
    """
    merge B to A
    """
    for key, value in b.items():
        if a.get(key) is not None:
            raise KeyError('key " {} " is already in dictionary!'.format(key))
        a[key] = value


def sound_tracking_data():
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    g1_data = generate_g1_data(handler, size_y=1, size_x=4)
    gru_data = generate_gru_data(handler, size_y=3, size_x=16)
    g3_data = generate_g3_data(handler, size_y=1, size_x=1)

    data_all = {}
    merge_dict(data_all, g1_data)
    merge_dict(data_all, gru_data)
    merge_dict(data_all, g3_data)

    return data_all
