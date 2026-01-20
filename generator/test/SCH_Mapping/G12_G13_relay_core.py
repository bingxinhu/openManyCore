import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen12_Gen13_relay_cores(phase_en, clock, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group13 Input Buffer Mapping
    chip (1, 1)
    (12, 9), (13, 9), (14, 9), (15, 9)

    chip (2, 2)
    (9, 0), (11, 0), (13, 0), (15, 0)
    """
    map_config = {
        'sim_clock': None,
            0:{
                'clock':None,
                0:{
                    'clock': clock,
                    'trigger':0,
                    'mode': 1
                }
            }
        }
    for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # relay 7*256
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=223, receive_num=0, t_mode=1, soma_in_en=0, relay_num=223)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # relay 7*256
    if phase_en[1] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=223, receive_num=0, t_mode=1, soma_in_en=0, relay_num=223)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # relay 7*256
    if phase_en[2] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=223, receive_num=0, t_mode=1, soma_in_en=0, relay_num=223)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # relay 15*256
    if phase_en[3] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=479, receive_num=0, t_mode=1, soma_in_en=0, relay_num=479)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 2 **************************************************#
    # relay 20*256
    if phase_en[4] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            temp_phase = phase
            rhead_base = 0x300
            if temp_phase > 0:
                while temp_phase > 0:
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                        break
                    else:
                        temp_phase -= 1
                if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=639, receive_num=0, t_mode=1, soma_in_en=0, relay_num=639)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 3 **************************************************#
    # relay 20*256
    if phase_en[5] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[5], L5_num=delay_L5[5], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            temp_phase = phase
            rhead_base = 0x300
            if temp_phase > 0:
                while temp_phase > 0:
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                        break
                    else:
                        temp_phase -= 1
                if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=639, receive_num=0, t_mode=1, soma_in_en=0, relay_num=639)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 4 **************************************************#
    # relay 22*256
    if phase_en[6] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 9), (13, 9), (14, 9), (15, 9), (5, 0), (7, 0), (9, 0), (11, 0)]:

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            temp_phase = phase
            rhead_base = 0x300
            if temp_phase > 0:
                while temp_phase > 0:
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                        break
                    else:
                        temp_phase -= 1
                if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=1, \
                         addr_din_base=0x380, addr_din_length=703, receive_num=0, t_mode=1, soma_in_en=0, relay_num=703)
            if core_y == 9:
                router.Nx = 16 + 4 + (core_x - 12) * 2 - core_x
                router.Ny = 0
            elif core_y == 0:
                router.Nx = 0
                router.Ny = -1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    return map_config