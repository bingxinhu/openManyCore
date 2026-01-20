import os
import sys
import platform

sys.path.append(os.getcwd())
from generator.detection.ObstacleNet.g0_check import check as g0_check
from generator.detection.ObstacleNet.g1_check import check as g1_check
from generator.detection.ObstacleNet.g2_check import check as g2_check
from generator.detection.ObstacleNet.g3_check import check as g3_check
from generator.detection.ObstacleNet.g4_check import check as g4_check
from generator.detection.ObstacleNet.g5_check import check as g5_check
from generator.detection.ObstacleNet.g6_check import check as g6_check
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.detection.ObstacleNet.data_gen import detection_data
from generator.detection.detection_data_handler import DetectionDataHandler

case_file_name = 'Det_SD_12'

# **** Net En ****
obstacle_en = 1
mouse_en = 0
axon_delay_empty_phase = 0

offset = 1 + axon_delay_empty_phase

# **** Group En ****
g0_en = 1
g1_en = 1
g2_en = 1
g3_en = 1
g4_en = 1
g5_en = 1
g6_en = 1

phase = None
step = 0

if 'Windows' in platform.platform():
    compare = ResultCompareWithClockSpecificSimulator(data_file_name=case_file_name, save_ref_data_en=True,
                                                      phase_en=phase,
                                                      print_matched=True, step=step)
else:
    compare = ResultCompare(data_2_file_name=case_file_name, save_data_1_en=True, phase_en=phase,
                            print_matched=True)

obstacle_handler = DetectionDataHandler(name='sd', pretrained=True)
obstacle_data_all = detection_data(handler=obstacle_handler) if obstacle_en else None
mouse_handler = DetectionDataHandler(name='mouse', pretrained=True)
mouse_data_all = detection_data(handler=mouse_handler) if mouse_en else None

if obstacle_en:
    if g0_en:
        g0_check(ref_data_obstacle=obstacle_data_all, ref_data_mouse=mouse_data_all, compare=compare, size_y=(3, 4),
                 size_x=(8, 16), chip=(0, 0), offset=1)

    if g1_en:
        g1_check(ref_data=obstacle_data_all, compare=compare, size_y=(4, 6), size_x=(8, 16), chip=(0, 0), offset=offset)

    if g2_en:
        g2_check(ref_data=obstacle_data_all, compare=compare, size_y=(6, 7), size_x=(8, 16), chip=(0, 0), offset=offset)

    if g3_en:
        g3_check(ref_data=obstacle_data_all, compare=compare, size_y=(7, 8), size_x=(8, 16), chip=(0, 0), offset=offset)

    if g4_en:
        g4_check(ref_data=obstacle_data_all, compare=compare, size_y=(8, 9), size_x=(12, 16), chip=(0, 0),
                 offset=offset)

    if g5_en:
        g5_check(ref_data=obstacle_data_all, compare=compare, size_y=(8, 9), size_x=(10, 12), chip=(0, 0),
                 offset=offset)

    if g6_en:
        g6_check(ref_data=obstacle_data_all, compare=compare, size_y=(8, 9), size_x=(9, 10), chip=(0, 0), offset=offset)

if mouse_en:
    if g1_en:
        g1_check(ref_data=mouse_data_all, compare=compare, size_y=(4, 6), size_x=(8 - 8, 16 - 8), chip=(0, 0),
                 offset=offset)
    if g2_en:
        g2_check(ref_data=mouse_data_all, compare=compare, size_y=(6, 7), size_x=(8 - 8, 16 - 8), chip=(0, 0),
                 offset=offset)
    if g3_en:
        g3_check(ref_data=mouse_data_all, compare=compare, size_y=(7, 8), size_x=(8 - 8, 16 - 8), chip=(0, 0),
                 offset=offset)
    if g4_en:
        g4_check(ref_data=mouse_data_all, compare=compare, size_y=(8, 9), size_x=(12 - 8, 16 - 8), chip=(0, 0),
                 offset=offset)
    if g5_en:
        g5_check(ref_data=mouse_data_all, compare=compare, size_y=(8, 9), size_x=(10 - 8, 12 - 8), chip=(0, 0),
                 offset=offset)
    if g6_en:
        g6_check(ref_data=mouse_data_all, compare=compare, size_y=(8, 9), size_x=(9 - 8, 10 - 8), chip=(0, 0),
                 offset=offset)

compare.run()
compare.show_result()
