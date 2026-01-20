


def add_one_core_prims(prims, old_core_prims):
    for i in range(len(old_core_prims['axon'])):
        prims.append({
            'axon': old_core_prims['axon'][i],
            'soma1': old_core_prims['soma1'][i],
            'router': old_core_prims['router'][i],
            'soma2': old_core_prims['soma2'][i]
        })



def changeIR(map, chip_x, chip_y, group_idx_list):
    map_config = {
        'sim_clock': None,
    }

    for group_idx in group_idx_list:
        for chip_x_idx in range(chip_x):
            for chip_y_idx in range(chip_y):
                for core_x in range(16):
                    for core_y in range(10):
                        if map[0][group_idx].get(((chip_x_idx, chip_y_idx), (core_x, core_y))) != None:    #该core存在
                            if map_config.get(((chip_x_idx, chip_y_idx), 0)) == None: #新的IR里没有，则添加
                                map_config[((chip_x_idx, chip_y_idx), 0)] = {
                                }
                            if map_config[((chip_x_idx, chip_y_idx), 0)].get(group_idx) == None:
                                map_config[((chip_x_idx, chip_y_idx), 0)][group_idx] = {
                                    'clock': map[0][group_idx]['clock'],
                                    'mode': 1,
                                    ((chip_x_idx, chip_y_idx), (core_x, core_y)): {
                                        'prims': []
                                    }
                                }
                            if map_config[((chip_x_idx, chip_y_idx), 0)][group_idx].get(((chip_x_idx, chip_y_idx), (core_x, core_y))) == None:
                                map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))] = {'prims': []}
                            # add_one_core_prims(map_config[((chip_x_idx, chip_y_idx), group_idx)][0][((chip_x_idx, chip_y_idx), (core_x, core_y))]['prims'],
                            #                    map[0][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))])
                            map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))]['prims'] = \
                                map[0][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))]['prims']
                            if map[0][group_idx][(((chip_x_idx, chip_y_idx), (core_x, core_y)))].get('instant_prims') != None:
                                map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))]['instant_prims'] = map[0][group_idx][(((chip_x_idx, chip_y_idx), (core_x, core_y)))]['instant_prims']
                            if map[0][group_idx][(((chip_x_idx, chip_y_idx), (core_x, core_y)))].get('registers') != None:
                                map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x, core_y))]['registers'] = map[0][group_idx][(((chip_x_idx, chip_y_idx), (core_x, core_y)))]['registers']

    return map_config