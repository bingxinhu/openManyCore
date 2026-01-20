from primitive.Prim_09_Router import Prim_09_Router

def get_real_core_idx(dst_x, dst_y, dst_chip_x, dst_chip_y):
    while dst_x > 15:
        dst_chip_x += 1
        dst_x -= 16
    while dst_x < 0:
        dst_chip_x -= 1
        dst_x += 16
    while dst_y > 9:
        dst_chip_y += 1
        dst_y -= 10
    while dst_y < 0:
        dst_chip_y -= 1
        dst_y += 10
    return dst_x, dst_y, dst_chip_x, dst_chip_y

def add_router_info(map_config, group_idx_list, chip_x_num, chip_y_num, core_x_num=16, core_y_num=10):
    """
    为map_config中的路由增加路由信息
    chip_x_num, chip_y_num  --  chip array
    core_x_num, core_y_num  --  core array
    涉及到组间同步的时候，只能自动完成两组在同一个phase数的时候组间传输的情况
    """
    for chip_x_idx in range(chip_x_num):
        for chip_y_idx in range(chip_y_num):
            for core_x_idx in range(core_x_num):
                for core_y_idx in range(core_y_num):
                    for group_idx in group_idx_list:
                        if map_config[0][group_idx].get(((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))) != None:
                            router_list = map_config[0][group_idx][((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['router']
                            for phase_idx in range(len(router_list)):
                                router = router_list[phase_idx]
                                if router != None:
                                    if router.Send_en == 1 and router.Receive_sign_en != 1:
                                        for rhead in router.RHeadList:
                                            if rhead['EN'] == 1:
                                                send_dst = {'data_num': rhead['pack_per_Rhead'] + 1,
                                                            'T_mode': rhead['T'],
                                                            'Rhead_num': 1,
                                                            'sync_en': 0}
                                                dst_core_list = []
                                                dst_x = core_x_idx + rhead['X']
                                                dst_y = core_y_idx + rhead['Y']
                                                dst_chip_x = chip_x_idx
                                                dst_chip_y = chip_y_idx
                                                dst_x, dst_y, dst_chip_x, dst_chip_y = get_real_core_idx(dst_x, dst_y,dst_chip_x, dst_chip_y)
                                                dst_core_list.append(((dst_chip_x, dst_chip_y), (dst_x, dst_y)))
                                                src_chip_x, src_chip_y, src_x, src_y = chip_x_idx, chip_y_idx, core_x_idx, core_y_idx
                                                src_group = group_idx
                                                if rhead['Q'] == 1: # 多播包
                                                    for dst_group_idx in group_idx_list:
                                                        if map_config[0][dst_group_idx].get(((dst_chip_x, dst_chip_y), (dst_x, dst_y))) != None:
                                                            if phase_idx < len(map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router']):
                                                                dst_router = map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router'][phase_idx]
                                                                if dst_router != None and dst_router.Receive_en == 1:
                                                                    if dst_group_idx != src_group:
                                                                        dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                                                                                'T_mode': rhead['T'],
                                                                                                                'Rhead_num': 1,
                                                                                                                'sync_en': 1,
                                                                                                                'sync_phase_num': phase_idx
                                                                                                                })
                                                                        send_dst['sync_en'] = 1
                                                                        send_dst['sync_phase_num'] = phase_idx
                                                                    else:
                                                                        dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                'data_num':rhead['pack_per_Rhead'] + 1,
                                                                                                                'T_mode': rhead['T'],
                                                                                                                'Rhead_num': 1,
                                                                                                                'sync_en': 0
                                                                                                                })

                                                                    while dst_router.CXY == 1:
                                                                        #src_chip_x, src_chip_y, src_x, src_y = dst_chip_x, dst_chip_y, dst_x, dst_y
                                                                        #src_group = dst_group_idx
                                                                        dst_x = dst_x + dst_router.Nx
                                                                        dst_y = dst_y + dst_router.Ny
                                                                        dst_x, dst_y, dst_chip_x, dst_chip_y = get_real_core_idx(dst_x, dst_y,dst_chip_x, dst_chip_y)
                                                                        dst_core_list.append(((dst_chip_x, dst_chip_y), (dst_x, dst_y)))
                                                                        for dst_group_idx in group_idx_list:
                                                                            if map_config[0][dst_group_idx].get(((dst_chip_x, dst_chip_y), (dst_x, dst_y))) != None:
                                                                                if phase_idx < len(map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router']):
                                                                                    dst_router = map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router'][phase_idx]
                                                                                    if dst_router.Receive_en == 1:
                                                                                        if dst_group_idx != group_idx:
                                                                                            dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                                    'data_num': rhead['pack_per_Rhead'] + 1,
                                                                                                                                    'T_mode': rhead['T'],
                                                                                                                                    'Rhead_num': 1,
                                                                                                                                    'sync_en': 1,
                                                                                                                                    'sync_phase_num': phase_idx
                                                                                                                                    })

                                                                                        else:
                                                                                            dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                                    'data_num': rhead['pack_per_Rhead'] + 1,
                                                                                                                                    'T_mode': rhead['T'],
                                                                                                                                    'Rhead_num': 1,
                                                                                                                                    'sync_en': 0
                                                                                                                                    })
                                                                                    # else:
                                                                                    #     raise Exception('({:d}, {:d}), ({:d}, {:d}) does not receive data while it should do!!'.format(dst_chip_x, dst_chip_y, dst_x, dst_y))
                                                                                break
                                                                            # else:
                                                                            #     raise Exception('({:d}, {:d}), ({:d}, {:d}) does not exist!!'.format(dst_chip_x, dst_chip_y, dst_x, dst_y))
                                                            break
                                                else:
                                                    for dst_group_idx in group_idx_list:
                                                        if map_config[0][dst_group_idx].get(((dst_chip_x, dst_chip_y), (dst_x, dst_y))) != None:
                                                            if phase_idx < len(map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router']):
                                                                dst_router = map_config[0][dst_group_idx][((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['router'][phase_idx]
                                                                if dst_router != None and dst_router.Receive_en == 1:
                                                                    if dst_group_idx != src_group:
                                                                        dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                                                                                'T_mode': rhead['T'],
                                                                                                                'Rhead_num': 1,
                                                                                                                'sync_en': 1,
                                                                                                                'sync_phase_num': phase_idx
                                                                                                                })
                                                                        send_dst['sync_en'] = 1
                                                                        send_dst['sync_phase_num'] = phase_idx
                                                                    else:
                                                                        dst_router.recv_source_core_grp.append({'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                                                                                'data_num':rhead['pack_per_Rhead'] + 1,
                                                                                                                'T_mode': rhead['T'],
                                                                                                                'Rhead_num': 1,
                                                                                                                'sync_en': 0
                                                                                                                })

                                                send_dst['core_id'] = dst_core_list
                                                router.send_destin_core_grp.append(send_dst)
                            break
    return map_config









