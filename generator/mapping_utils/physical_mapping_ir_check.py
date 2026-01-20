import json
import warnings
import sys


def check_global(table: dict):
    array_config = table['array_config']
    assert int(array_config['chip_w']) > 0
    assert int(array_config['chip_h']) > 0
    assert int(array_config['core_w']) > 0
    assert int(array_config['core_h']) > 0
    if array_config['core_w'] != 16 or array_config['core_h'] != 10:
        warnings.warn('core array config is not 16 * 10, which is different from the Chip!')
    assert int(table['ROUTE_STRATEGY']) in [0, 1]
    assert int(table['optimize_goal']) in [0]  # temporarily
    assert int(table['SYS_CLK_DIV']) in [0, 1, 3]
    assert int(table['SERDES_CLK_DIV']) in [0, 1, 3, 5, 7]


def physical_mapping_ir_check(file_dir='', position_check=True):
    core_info_sta = {}  # ((chip_x, chip_y), (chip_x, chip_y)) : group_idx
    trigger_info_sta = {}  # group_idx : trigger
    send_dst_core = set()
    with open(file_dir, 'r') as f:
        table = json.load(f)
        check_global(table)  # check global config
        phase_list = table['phase_list']
        phase_idx = 0
        while True:
            if phase_list.get(str(phase_idx)) is None:
                for i in range(phase_idx, 32):
                    assert phase_list.get(str(i)) is None
                break
            for group_idx, group in phase_list[str(phase_idx)].items():
                for core in group:
                    if position_check:
                        assert int(core['old_core_id']['chip_x']) == int(core['core_id']['chip_x'])
                        assert int(core['old_core_id']['chip_y']) == int(core['core_id']['chip_y'])
                        assert int(core['old_core_id']['core_x']) == int(core['core_id']['core_x'])
                        assert int(core['old_core_id']['core_y']) == int(core['core_id']['core_y'])
                    assert int(core['old_core_id']['chip_x']) >= 0
                    assert int(core['old_core_id']['chip_y']) >= 0
                    assert int(core['core_id']['chip_x']) >= 0
                    assert int(core['core_id']['chip_y']) >= 0
                    assert int(core['old_core_id']['core_x']) >= 0
                    assert int(core['old_core_id']['core_y']) >= 0
                    assert int(core['core_id']['core_x']) >= 0
                    assert int(core['core_id']['core_y']) >= 0
                    assert int(core['trigger']) in [0, 1, 2, 3]
                    assert int(core['A2S2_mode']) in [0, 1]
                    assert int(core['A_time']) >= 0
                    assert int(core['S1_time']) >= 0
                    assert int(core['S2_time']) >= 0
                    assert core['fixed_position'] in [True, False]
                    assert core['multicast_core'] in [True, False]
                    assert int(core['multicast_dst']['chip_x']) >= 0
                    assert int(core['multicast_dst']['chip_y']) >= 0
                    assert int(core['multicast_dst']['core_x']) >= 0
                    assert int(core['multicast_dst']['core_y']) >= 0
                    if core['multicast_core']:
                        send_dst_core.add(
                            ((int(core['multicast_dst']['chip_x']), int(core['multicast_dst']['chip_y'])),
                             (int(core['multicast_dst']['core_x']), int(core['multicast_dst']['core_y']))))
                    for packet in core['packets']:
                        assert int(packet['dst_core_id']['chip_x']) >= 0
                        assert int(packet['dst_core_id']['chip_y']) >= 0
                        assert int(packet['dst_core_id']['core_x']) >= 0
                        assert int(packet['dst_core_id']['core_y']) >= 0
                        send_dst_core.add(
                            ((int(packet['dst_core_id']['chip_x']), int(packet['dst_core_id']['chip_y'])),
                             (int(packet['dst_core_id']['core_x']), int(packet['dst_core_id']['core_y']))))
                        assert int(packet['pack_num']) > 0
                        assert packet['multicast_pack'] in [True, False]
                    #
                    chip_x, chip_y = int(core['core_id']['chip_x']), int(core['core_id']['chip_y'])
                    core_x, core_y = int(core['core_id']['core_x']), int(core['core_id']['core_y'])
                    trigger = int(core['trigger'])
                    if trigger_info_sta.get(group_idx) is None:
                        trigger_info_sta[group_idx] = trigger
                    else:
                        assert trigger == trigger_info_sta[group_idx]
                    if phase_idx == 0:
                        assert core_info_sta.get(((chip_x, chip_y), (core_x, core_y))) is None
                        core_info_sta[((chip_x, chip_y), (core_x, core_y))] = (trigger, group_idx)
                    else:
                        assert core_info_sta[((chip_x, chip_y), (core_x, core_y))] == (trigger, group_idx)
            phase_idx += 1
            if phase_idx > 31:
                break
        all_core = set(i for i in core_info_sta.keys())
        assert len(send_dst_core.difference(all_core)) == 0


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Error: No File is Selected, Please specific the File to be checked!")
    else:
        for file in sys.argv[1:]:
            print("--- Start Checking {:s}".format(file))
            physical_mapping_ir_check(file_dir=file)
            print("\t Correct!")
