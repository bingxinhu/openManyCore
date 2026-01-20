# generate router case according to json
# 2022.5.21 sch
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.prims import p06, p09, p41
from itertools import product
import json
import os
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
from math import ceil


def reduce_num(num_in, max1=256, max2=4096):
    for r1 in range(1, max1 + 1):
        if num_in % r1 == 0 and 1 <= num_in // r1 <= max2:
            return True, r1, num_in // r1
    return False, 0, 0


def gen_router_case(file_dir='', sim_clock=0, clock_in_phase=1048575):
    map_config = {
        'sim_clock': sim_clock,
        # (chip, 0): {
        #     0: {
        #         'clock': clock_in_phase,
        #         'mode': 1,
        #     }
        # }
    }
    with open(file_dir, 'r') as f:
        table = json.load(f)
        if table.get('chip_w') is None:
            chip_w, chip_h = 3, 3
            core_w, core_h = 16, 10
        else:
            chip_w, chip_h = table['chip_w'], table['chip_h']
            core_w, core_h = table['core_w'], table['core_h']
        assert chip_h > 0 and chip_w > 0
        assert 0 < core_w <= 16 and 0 < core_h <= 10
        group_cnt_each_chip = dict()
        group_idx_mapping = dict()
        group_idx_each_core = dict()
        recv_pack_num_each_core = dict()  # (normal, multicast)
        recv_num_each_core = dict()  #
        for c_w, c_h in product(range(chip_w), range(chip_h)):
            group_cnt_each_chip[(c_w, c_h)] = 0
            map_config[((c_w, c_h), 0)] = {}
        # check if a group has cores in multi-chip
        for phase_idx, phase in table['phase_list'].items():
            for grp_idx, group in phase.items():
                chip_id = (group[0]['core_id']['chip_x'], group[0]['core_id']['chip_y'])
                group_idx_mapping[grp_idx] = group_cnt_each_chip[chip_id]
                group_cnt_each_chip[chip_id] += 1
                for core in group:
                    assert chip_id == (core['core_id']['chip_x'], core['core_id']['chip_y']), \
                        'one group must be in same chip'
            break
        # router.CXY
        for phase_idx, phase in table['phase_list'].items():
            for grp_idx, group in phase.items():
                real_grp_idx = group_idx_mapping[grp_idx]
                for core in group:
                    if core['A2S2_mode'] == 1:
                        axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1,
                                   addr_ina=0x10000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10000 >> 2,
                                   addr_out=0x10000 >> 2, axon_delay=True,
                                   L4_num=core['A_l4'], L5_num=core['A_l5'], A2S2_mode=True)
                        # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                    else:
                        axon = None
                    chip_id = (core['core_id']['chip_x'], core['core_id']['chip_y'])
                    core_id = (core['core_id']['core_x'], core['core_id']['core_y'])
                    if int(phase_idx) == 0:
                        assert group_idx_each_core.get((chip_id, core_id)) is None
                    else:
                        assert group_idx_each_core.get((chip_id, core_id)) is not None
                    group_idx_each_core[(chip_id, core_id)] = real_grp_idx
                    if recv_pack_num_each_core.get((phase_idx, chip_id, core_id)) is None:
                        recv_pack_num_each_core[(phase_idx, chip_id, core_id)] = [0, 1600]
                    if recv_num_each_core.get((phase_idx, chip_id, core_id)) is None:
                        recv_num_each_core[(phase_idx, chip_id, core_id)] = 0
                    if map_config[(chip_id, 0)].get(real_grp_idx) is None:
                        assert int(phase_idx) == 0
                        map_config[(chip_id, 0)][real_grp_idx] = {'clock': clock_in_phase, 'mode': 1}
                    if map_config[(chip_id, 0)][real_grp_idx].get((chip_id, core_id)) is None:
                        assert int(phase_idx) == 0
                        map_config[(chip_id, 0)][real_grp_idx][(chip_id, core_id)] = {'prims': []}
                    prims_list = map_config[(chip_id, 0)][real_grp_idx][(chip_id, core_id)]['prims']
                    if core['multicast_core'] == 1:
                        dst_chip = (core['multicast_dst']['chip_x'], core['multicast_dst']['chip_y'])
                        dst_core = (core['multicast_dst']['core_x'], core['multicast_dst']['core_y'])
                        dx = (dst_chip[0] - chip_id[0]) * 16 + dst_core[0] - core_id[0]
                        dy = (dst_chip[1] - chip_id[1]) * 10 + dst_core[1] - core_id[1]
                    else:
                        dx, dy = 0, 0
                    assert -128 <= dx <= 127 and -128 <= dy <= 127, 'dst_core is too far to reach!'
                    router_prim = p09(rhead_mode=1, send_en=0, receive_en=0, send_num=0, receive_num=0,
                                      addr_din_base=0x380, addr_din_length=1599, addr_rhead_base=0x300,
                                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                                      cxy=core['multicast_core'], relay_num=0, nx=dx, ny=dy, data_in=None)
                    prims_list.append({'axon': axon, 'soma1': None, 'router': router_prim, 'soma2': None})
        # router.send
        for phase_idx, phase in table['phase_list'].items():
            for grp_idx, group in phase.items():
                real_grp_idx = group_idx_mapping[grp_idx]
                for core in group:
                    chip_id = (core['core_id']['chip_x'], core['core_id']['chip_y'])
                    core_id = (core['core_id']['core_x'], core['core_id']['core_y'])
                    if recv_pack_num_each_core.get((phase_idx, chip_id, core_id)) is None:
                        recv_pack_num_each_core[(phase_idx, chip_id, core_id)] = [0, 1600]
                    if recv_num_each_core.get((phase_idx, chip_id, core_id)) is None:
                        recv_num_each_core[(phase_idx, chip_id, core_id)] = 0
                    if map_config[(chip_id, 0)].get(real_grp_idx) is None:
                        assert int(phase_idx) == 0
                        map_config[(chip_id, 0)][real_grp_idx] = {'clock': clock_in_phase, 'mode': 1}
                    if map_config[(chip_id, 0)][real_grp_idx].get((chip_id, core_id)) is None:
                        assert int(phase_idx) == 0
                        map_config[(chip_id, 0)][real_grp_idx][(chip_id, core_id)] = {'prims': []}
                    prims_list = map_config[(chip_id, 0)][real_grp_idx][(chip_id, core_id)]['prims']
                    router_prim = prims_list[int(phase_idx)]['router']
                    router_prim.Addr_Rhead_base = MapConfigGen.get_router_rhead_base(prims_list, index=int(phase_idx),
                                                                                     limit=0x380)
                    for packet in core['packets']:
                        dst_chip = (packet['dst_core_id']['chip_x'], packet['dst_core_id']['chip_y'])
                        dst_core = (packet['dst_core_id']['core_x'], packet['dst_core_id']['core_y'])
                        if recv_pack_num_each_core.get((phase_idx, dst_chip, dst_core)) is None:
                            recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)] = [0, 1600]
                        if recv_num_each_core.get((phase_idx, dst_chip, dst_core)) is None:
                            recv_num_each_core[(phase_idx, dst_chip, dst_core)] = 0
                        dx = (dst_chip[0] - chip_id[0]) * 16 + dst_core[0] - core_id[0]
                        dy = (dst_chip[1] - chip_id[1]) * 10 + dst_core[1] - core_id[1]
                        assert -128 <= dx <= 127 and -128 <= dy <= 127, 'dst_core is too far to reach!'
                        router_prim.Send_en = 1
                        router_prim.Send_number += packet['pack_num']
                        router_prim.Addr_Rhead_length += 1
                        if packet['multicast_pack']:
                            a_base = recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][1] - packet['pack_num']
                            while True:  # multicast core
                                m_router_prim = map_config[
                                    (dst_chip, 0)][group_idx_each_core[(dst_chip, dst_core)]][
                                    (dst_chip, dst_core)]['prims'][int(phase_idx)]['router']
                                recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][1] -= packet['pack_num']
                                assert recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][1] >= \
                                       recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][0]
                                recv_num_each_core[(phase_idx, dst_chip, dst_core)] += 1
                                if m_router_prim.CXY == 0:
                                    break
                                else:
                                    m_router_prim.Relay_number += packet['pack_num']
                                    dst_x, dst_y, dst_chip_x, dst_chip_y = MapConfigGen._get_real_core_idx(
                                        dst_core[0] + m_router_prim.Nx, dst_core[1] + m_router_prim.Ny, dst_chip[0],
                                        dst_chip[1])
                                    dst_chip = (dst_chip_x, dst_chip_y)
                                    dst_core = (dst_x, dst_y)
                        else:
                            recv_num_each_core[(phase_idx, dst_chip, dst_core)] += 1
                            a_base = recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][0]
                            recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][0] += packet['pack_num']
                            assert recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][1] >= \
                                   recv_pack_num_each_core[(phase_idx, dst_chip, dst_core)][0]
                        router_prim.addRHead(S=0, T=1, P=0, Q=1 if packet['multicast_pack'] else 0,
                                             X=dx, Y=dy, A=a_base, pack_per_Rhead=packet['pack_num'] - 1,
                                             A_offset=0, Const=0, EN=1)
                    if router_prim.Send_en == 1:
                        router_prim.Send_number -= 1
                        router_prim.Addr_Rhead_length = ceil(router_prim.Addr_Rhead_length / 2) - 1
    # soma1 router.recv
    for phase_idx, phase in table['phase_list'].items():
        for grp_idx, group in phase.items():
            real_grp_idx = group_idx_mapping[grp_idx]
            for core in group:
                chip_id = (core['core_id']['chip_x'], core['core_id']['chip_y'])
                core_id = (core['core_id']['core_x'], core['core_id']['core_y'])
                prims = map_config[(chip_id, 0)][real_grp_idx][(chip_id, core_id)]['prims'][int(phase_idx)]
                # router.recv
                router_prim = prims['router']
                if router_prim.CXY == 1:
                    router_prim.Relay_number -= 1
                if recv_pack_num_each_core[(phase_idx, chip_id, core_id)][0] > 0 or \
                        recv_pack_num_each_core[(phase_idx, chip_id, core_id)][1] < 1600:
                    router_prim.Receive_en = 1
                    router_prim.Receive_number = recv_num_each_core[(phase_idx, chip_id, core_id)] - 1
                # soma1
                if router_prim.Send_en == 1:
                    data_size = router_prim.Send_number + 1
                    assert data_size % 2 == 0, 'the smallest block which soma can operate is 16Byte!'
                    data_size //= 2
                    res = reduce_num(data_size)
                    assert res[0], 'soma1\'s data size wrong!'
                    soma1_prim = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x0000, length_in=res[1] * 16,
                                     length_out=res[1] * 16, length_ciso=1, num_in=res[2], num_ciso=res[2],
                                     num_out=res[2], type_in=1, type_out=1, in_cut_start=0, in_row_max=0,
                                     row_ck_on=0, data_in=[], data_ciso=None)
                    prims['soma1'] = soma1_prim
    return map_config


def run_simulator(file_name='', sim_clock=20000):
    case_name = file_name.split('.')[0].replace('/', '_')

    map_config = gen_router_case(file_dir='./temp/router/' + file_name, sim_clock=sim_clock)
    MapConfigGen.add_router_info(map_config)

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_name + "\\"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        del_command = 'rd/s/q cmp_out'
        os.system(del_command)
        os.chdir(c_path)

    test_config = {
        'tb_name': case_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.open_burst.dict,
        'test_group_phase': [(0, 1)]
    }
    tester = TestEngine(map_config, test_config)
    if tester.run_test():
        print('Simulation Success!')
    else:
        print('Simulation Failed!')


if __name__ == '__main__':
    run_simulator(file_name='Q00425.json', sim_clock=350000)
