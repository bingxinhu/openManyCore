from generator.detection.detection_data_handler import DetectionDataHandler
from generator.detection.ObstacleNet.data_gen import detection_data
import torch
import cv2
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.sound_tracking_dynamic.data_gen import sound_tracking_data
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler
import numpy as np
from PIL import Image


def detection_ref_model(picture_name, net_name, pretrained):
    handler = DetectionDataHandler(name=net_name, pretrained=pretrained)

    img = cv2.imread(picture_name, cv2.IMREAD_UNCHANGED)
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    _ = handler.model(x)
    data_all = detection_data(handler=handler)

    # x = data_all['conv1']['input'][(2, 0)]

    y = data_all['avgpool']['output'][(0, 0)]

    index = picture_name.split('/')[-1].split('.')[0]

    out = ResultCompareWithClockSpecificSimulator.list2file(y, data_type=0)
    out_hex = np.array([int(item[:-1], 16) for item in out]).astype(np.int32).tolist()
    out_str = [str(item) + '\n' for item in out_hex]
    with open('temp/ref_data/' + index + net_name + 'output_v2_dec.txt', 'w') as f:
        f.writelines(out_str)


def st_ref_model(picture_name):
    handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
    # img = Image.open('./temp/sound_tracking/sound_mat1.png')
    # xx = np.fromfile(picture_name, dtype=np.uint8)
    # yy = cv2.imdecode(xx, -1)
    # img = yy

    img = cv2.imread(picture_name, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.int8).reshape((257, 41, 16))[:, :, :8]
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    _ = handler.model(x)
    data_all = sound_tracking_data(handler=handler)
    # y = data_all['conv1']['input'][(0, 0)]
    y = data_all['g3_output'][(0, 0)]
    out = ResultCompareWithClockSpecificSimulator.list2file(y, data_type=1)
    return out


if __name__ == '__main__':
    # detection_ref_model(picture_name='temp/obstacle/img_aftpre00005.png', net_name='mouse', pretrained=True)
    # # for i in range(1):
    # #     # name = 'sound{:d}.png'.format(i + 1)
    # #     result = st_ref_model(picture_name='./temp/sound_tracking/sound_mat1.png')
    # #     with open('temp/sound_tracking/result---{:d}.txt'.format(i + 1), 'w') as f:
    # #         f.writelines(result)
    # result = st_ref_model(picture_name='./sjr/soundRecorded1.png')
    # with open('sjr/soundRecorded1_result.txt', 'w') as f:
    #     f.writelines(result)
    # for i in range(1):
    #     name = 'picRecorded{:d}.png'.format(i + 1)
    #     detection_ref_model(picture_name='temp/picLoopCheck/' + name, net_name='obstacle', pretrained=True)
    #     # detection_ref_model(picture_name='temp/picLoopCheck/' + name, net_name='mouse', pretrained=True)
    #     print(i)
    # detection_ref_model(picture_name='temp/obstacle/img_aftpre00005.png', net_name='obstacle', pretrained=True)
    detection_ref_model(picture_name='temp/picLoopCheck/picRecorded0_v2.png', net_name='obstacle', pretrained=True)
