

def change_ir_to_prims(map_config, group_idx_list, chip_x_num, chip_y_num, core_x_num, core_y_num):
    """
    为map_config中的路由增加路由信息
    chip_x_num, chip_y_num  --  chip array
    core_x_num, core_y_num  --  core array
    涉及到组间同步的时候，只能自动完成两组在同一个phase数的时候组间传输的情况
    """
    map = {
        'sim_clock': map_config['sim_clock'],
        0: {  # step group id
            "cycles_number": 1,
        }
    }
    for i in group_idx_list:
        map[0][i] = {
            'clock': None,
            'mode': 1,
        }
    for chip_x in range(chip_x_num):
        for chip_y in range(chip_y_num):
            for core_x in range(core_x_num):
                for core_y in range(core_y_num):
                    for group_idx in group_idx_list:
                        if map_config[0][group_idx].get(((chip_x, chip_y), (core_x, core_y))) != None and len(map_config[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))]['axon']) > 0:
                            if map[0][group_idx]['clock'] == None:
                                map[0][group_idx]['clock'] = map_config[0][group_idx]['clock']
                            axon = map_config[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['axon']
                            soma1 = map_config[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['soma1']
                            router = map_config[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['router']
                            soma2 = map_config[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['soma2']
                            if map[0][group_idx].get(((chip_x, chip_y), (core_x, core_y))) == None:
                                map[0][group_idx][((chip_x, chip_y), (core_x, core_y))] = {}
                                map[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['prims'] = []
                            for j in range(len(axon)):
                                map[0][group_idx][((chip_x, chip_y), (core_x, core_y))]['prims'].append({
                                    'axon': axon[j],
                                    'soma1': soma1[j],
                                    'router': router[j],
                                    'soma2': soma2[j]
                                })
                            if map_config[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))].get('instant_prims') != None:
                                map[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))]['instant_prims'] = map_config[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))]['instant_prims']
                            if map_config[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))].get('registers') != None:
                                map[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))]['registers'] = map_config[0][group_idx][(((chip_x, chip_y), (core_x, core_y)))]['registers']

    for group_idx in group_idx_list:
        if len(map[0][group_idx]) == 2:
            map[0].pop(group_idx)
    return map









