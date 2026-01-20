import os
import sys

sys.path.append(os.getcwd())
from generator.sound_tracking_dynamic.g0_check import check as g0_check
from generator.sound_tracking_dynamic.g1_check import check as g1_check
from generator.sound_tracking_dynamic.gru_check import check as gru_check
from generator.sound_tracking_dynamic.g3_check import check as g3_check
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.sound_tracking_dynamic.data_gen import sound_tracking_data
from generator.sound_tracking_dynamic.sound_tracking_data_handler import SoundTrackingDataHandler

case_file_name = 'ST_D01'

# **** Group En ****
g0_en = 1
g1_en = 1
gru_en = 1
g3_en = 1

phase = None
step = 41
empty_num = 1

data_all = sound_tracking_data()
handler = SoundTrackingDataHandler(input_channels=8, output_channels=16, hidden_size=128, sequence_length=39)
compare = ResultCompareWithClockSpecificSimulator(data_file_name=case_file_name, save_ref_data_en=True, phase_en=phase,
                                                  print_matched=True, step=step)

if g0_en:
    g0_check(ref_data=data_all, compare=compare, size_y=(0, 1), size_x=(12, 15), chip=(0, 0), empty_num=empty_num)

if g1_en:
    g1_check(ref_data=data_all, compare=compare, size_y=(0, 1), size_x=(0, 4), chip=(0, 0), empty_num=empty_num)

if gru_en:
    gru_check(ref_data=data_all, compare=compare, size_y=(1, 4), size_x=(0, 16), chip=(0, 0),
              sequence_length=handler.sequence_length, empty_num=empty_num)

if g3_en:
    g3_check(ref_data=data_all, compare=compare, size_y=(3, 4), size_x=(7, 8), chip=(0, 0), empty_num=empty_num)

compare.run()
compare.show_result()
