import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02
from primitive.Prim_09_Router import Prim_09_Router

def Gen_G9_shortcut_Map_Config(phase_en, clock, M, N, start_row=0, in_data_en=0, delay_L4=(0, 0, 0, 0, 0, 0), delay_L5=(0, 0, 0, 0, 0, 0)):
    """
    ResNet-50 Group9 Mapping
    with 32 function cores
    core array : 2 * 8
    """
    core_num = M * N
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
    for core in range(start_row * M, start_row * M + core_num):
        core_x = core % M
        core_y = core // M
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # 接收数据 20 * 128
    if phase_en[0] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    router.Receive_en = 1
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_y == 0:
                        router.Nx = 0
                        router.Ny = 1
                    else:
                        if core_x == 3:
                            router.Nx = -3
                            router.Ny = 1
                        elif core_x == 4:
                            router.Nx = -3
                            router.Ny = 2
                        elif core_x == 5:
                            router.Nx = -3
                            router.Ny = 3
                        elif core_x == 6:
                            router.Nx = -3
                            router.Ny = 4
            if core_x in [3, 4, 5, 6]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=128, \
                                cout=128, px=10, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19e00>>2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 2 **************************************************#
    # 接收数据 20 * 128
    if phase_en[1] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    router.Receive_en = 1
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_y == 0:
                        router.Nx = 0
                        router.Ny = 1
                    else:
                        if core_x == 3:
                            router.Nx = -3
                            router.Ny = 1
                        elif core_x == 4:
                            router.Nx = -3
                            router.Ny = 2
                        elif core_x == 5:
                            router.Nx = -3
                            router.Ny = 3
                        elif core_x == 6:
                            router.Nx = -3
                            router.Ny = 4
            if core_x in [3, 4, 5, 6]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=128, \
                                cout=128, px=10, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x19e00+20*128)>>2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 3 **************************************************#
    # 接收数据 20 * 128

    if phase_en[2] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    router.Receive_en = 1
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_y == 0:
                        router.Nx = 0
                        router.Ny = 1
                    else:
                        if core_x == 3:
                            router.Nx = -3
                            router.Ny = 1
                        elif core_x == 4:
                            router.Nx = -3
                            router.Ny = 2
                        elif core_x == 5:
                            router.Nx = -3
                            router.Ny = 3
                        elif core_x == 6:
                            router.Nx = -3
                            router.Ny = 4
            if core_x in [3, 4, 5, 6]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=128, \
                                cout=128, px=10, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x19e00+40*128)>>2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 4 **************************************************#
    # 接收数据 38 * 128

    if phase_en[3] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=607, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    router.Receive_en = 1
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_y == 0:
                        router.Nx = 0
                        router.Ny = 1
                    else:
                        if core_x == 3:
                            router.Nx = -3
                            router.Ny = 1
                        elif core_x == 4:
                            router.Nx = -3
                            router.Ny = 2
                        elif core_x == 5:
                            router.Nx = -3
                            router.Ny = 3
                        elif core_x == 6:
                            router.Nx = -3
                            router.Ny = 4
            if core_x in [3, 4, 5, 6]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=128, \
                                cout=128, px=10, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x19e00+60*128)>>2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 5 **************************************************#
    # 接收数据 98*128
    if phase_en[4] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=199, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1567, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    router.Receive_en = 1
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_y == 0:
                        router.Nx = 0
                        router.Ny = 1
                    else:
                        if core_x == 3:
                            router.Nx = -3
                            router.Ny = 1
                        elif core_x == 4:
                            router.Nx = -3
                            router.Ny = 2
                        elif core_x == 5:
                            router.Nx = -3
                            router.Ny = 3
                        elif core_x == 6:
                            router.Nx = -3
                            router.Ny = 4
            if core_x in [3, 4, 5, 6]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            if in_data_en == 1:
                if core_x in [3, 4, 5, 6]:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=128, \
                                cout=128, px=49, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x1cf00>>2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 6 **************************************************#
    # 整理数据 25 * 512  -- 1-1
    type = 0    # 发送数据的core_y
    if phase_en[5] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 25 * 128
                if core_y == 0:
                    addr_in = 0x1cf00 >> 2
                else:
                    addr_in = 0x19e00 >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=399, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 7 **************************************************#
    # 整理数据 25 * 512  -- 1-2
    type = 1  # 发送数据的core_y
    if phase_en[6] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 25 * 128
                if core_y == 0:
                    addr_in = 0x1cf00 >> 2
                else:
                    addr_in = 0x19e00 >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=399, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 8 **************************************************#
    # 整理数据 25 * 512  -- 2-1
    type = 0
    if phase_en[7] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 25 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 25 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 25 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=399, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 25 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 9 **************************************************#
    # 整理数据 25 * 512  -- 2-2
    type = 1
    if phase_en[8] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 25 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 25 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 25 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=399, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 25 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 10 **************************************************#
    # 整理数据 24 * 512  -- 3-1
    type = 0
    if phase_en[9] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 24 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 50 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 50 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=383, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 50 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 11 **************************************************#
    # 整理数据 24 * 512  -- 3-2
    type = 1
    if phase_en[10] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 24 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 50 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 50 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=383, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 50 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 9 **************************************************#
    # 整理数据 24 * 512  -- 4-1
    type = 0
    if phase_en[11] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 24 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 74 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 74 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=383, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 74 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 9 **************************************************#
    # 整理数据 24 * 512  -- 4-2
    type = 1
    if phase_en[12] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_x in [3, 4, 5, 6] and core_y == type:
                length = 24 * 128
                if core_y == 0:
                    addr_in = (0x1cf00 + 74 * 128) >> 2
                else:
                    addr_in = (0x19e00 + 74 * 128) >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                            addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_1_soma1_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0},
                ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            else:
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=383, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=3, t_mode=1, soma_in_en=0)
            if core_y != type:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Nx = 1
                    router.Ny = 0
                    router.Relay_number = router.Addr_Din_length
            if core_x in [3, 4, 5, 6] and core_y == type:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 1:
                    dst_x = 0
                    dst_y = 0
                else:
                    dst_x = 0
                    dst_y = 1
                A = (core_x - 3) * 16
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y != type:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 74 * 512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 10 **************************************************#
    # # shortcut卷积 - 14*7*512 -> 14*7*128 0x8380  1/6 -- 16
    if phase_en[13] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=2, py=8, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)
            a_data = p41(core_x=core_x, core_y=core_y, px=14, py=7, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)
            a = a_data.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_18_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': 512 * 128 * 1 * 1 // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[5] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_18_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': a_data.cin * a_data.Input_fm_Py * a_data.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                        cout=128, px=axon.Input_fm_Px, py=axon.Input_fm_Py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, in_cut_start=5, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x8380, in_row_max=8)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 14 **************************************************#
    # # shortcut卷积 - 14*7*512 -> 14*7*128 0x8380  2/6 -- 32
    if phase_en[14] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=2, py=16, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=(0x10000+16*512) >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                        cout=128, px=axon.Input_fm_Px, py=axon.Input_fm_Py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, in_cut_start=5, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x8380 + (16*128 >> 2), in_row_max=8)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 16 **************************************************#
    # # shortcut卷积 - 14*7*512 -> 14*7*128 0x8380  2/6 -- 32
    if phase_en[15] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=2, py=16, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=(0x10000+48*512) >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                        cout=128, px=axon.Input_fm_Px, py=axon.Input_fm_Py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, in_cut_start=5, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x8380 + (48*128 >> 2), in_row_max=8)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 17 **************************************************#
    # # None
    if phase_en[16] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 18 **************************************************#
    # # None
    if phase_en[17] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 19 **************************************************#
    # # None
    if phase_en[18] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 20 **************************************************#
    # # None
    if phase_en[19] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 21 **************************************************#
    # # shortcut卷积 - 14*7*512 -> 14*7*128 0x8380  1/12 -- 8
    if phase_en[20] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=2, py=4, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=(0x10000+80*512) >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                        cout=128, px=axon.Input_fm_Px, py=axon.Input_fm_Py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, in_cut_start=5, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x8380 + (80*128 >> 2), in_row_max=4)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 22 **************************************************#
    # # None
    if phase_en[21] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 23 **************************************************#
    # # None
    if phase_en[22] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 24 **************************************************#
    # # None
    if phase_en[23] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 21 **************************************************#
    # # shortcut卷积 - 14*7*512 -> 14*7*128 0x8380  1/12 -- 8
    if phase_en[24] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=2, py=4, pad_on=False, load_bias=0, cin=512, cout=128, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=(0x10000 + 88*512) >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                        cout=128, px=axon.Input_fm_Px, py=axon.Input_fm_Py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, in_cut_start=5, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x8380 + (88*128 >> 2), in_row_max=4)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 26 **************************************************#
    # # 发送
    if phase_en[25] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
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
            Router_Prim_1 = Prim_09_Router()
            Router_Prim_1.Rhead_mode = 1
            Router_Prim_1.CXY = 0b00
            Router_Prim_1.Send_en = 1
            Router_Prim_1.Receive_en = 0
            Router_Prim_1.Addr_Dout_base = 0x380
            Router_Prim_1.Dout_Mem_sel = 0
            Router_Prim_1.Addr_Dout_length = 783
            Router_Prim_1.Send_number = 1567
            Router_Prim_1.Addr_Rhead_base = rhead_base
            Router_Prim_1.Addr_Rhead_length = 1
            Router_Prim_1.Addr_Din_base = 0x800
            Router_Prim_1.Addr_Din_length = 0
            Router_Prim_1.Receive_number = 0
            Router_Prim_1.Nx = 0
            Router_Prim_1.Ny = 0
            Router_Prim_1.Send_PI_en = 0
            Router_Prim_1.Back_sign_en = 0
            Router_Prim_1.Send_PI_num = 0
            Router_Prim_1.Receive_sign_num = 0
            Router_Prim_1.Receive_PI_addr_base = 0
            Router_Prim_1.Relay_number = 0
            Router_Prim_1.Q = 0
            Router_Prim_1.Receive_sign_en = 0
            Router_Prim_1.T_mode = 1
            Router_Prim_1.Soma_in_en = 0

            dst_y = core_x // 2 + 2
            if core_y == 0:
                A = 0
            else:
                A = 7 * 14 * 32 // 8

            if core_x % 2 == 0:
                dst_x = 0
            else:
                dst_x = 4

            Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 0, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
            Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 1, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
            Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 2, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)
            Router_Prim_1.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x + 3, Y=dst_y - core_y, A=A, pack_per_Rhead=391, A_offset=0, Const=0, EN=1)

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(Router_Prim_1)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 27 **************************************************#
    # # None
    if phase_en[26] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    return map_config


test = False
if test:
    # ---------------------------------------------------------------------------------------#
    import os
    import sys
    case_file_name = 'M99999'
    c_path = os.getcwd()
    out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        del_command = 'rd/s/q cmp_out'
        os.system(del_command)
        os.chdir(c_path)
    # 真正执行的代码
    phase_en = np.zeros(32).astype(int)
    # 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
    # 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
    # group间
    phase_en[5:13] = 1     # 组间 5- 13
    # phase_en[1] = 0     # 组间
    #
    # phase_en[2] = 0     #
    # phase_en[3] = 1     #
    # phase_en[4] = 1     #
    # phase_en[5] = 1     #
    # phase_en[6] = 1  #
    # phase_en[7] = 1  #
    # phase_en[8] = 1  #
    # #
    # phase_en[9] = 1  #

    run = True  # 只生成map_config 还是生成后直接运行

    map_config = Gen_G9_shortcut_Map_Config(phase_en, 80000, M=8, N=2)

    from generator.test.SCH_Mapping.AddRouterInfo import add_router_info
    map_config = add_router_info(map_config=map_config, group_idx_list=[0], chip_x_num=1, chip_y_num=1, core_x_num=16, core_y_num=10)
    from generator.test.SCH_Mapping.changeIR_to_prims import change_ir_to_prims
    map_config = change_ir_to_prims(map_config=map_config, group_idx_list=[0], chip_x_num=1, chip_y_num=1, core_x_num=16, core_y_num=10)

    from generator.test.SCH_Mapping.changeIR import changeIR

    map_config = changeIR(map=map_config, chip_x=1, chip_y=1, group_idx_list=[0])

    map_config['sim_clock'] = min(400000, sum(phase_en) * 50000)
    import pickle
    with open('sch_0826_group9_16cores', 'wb') as f:
       pickle.dump(map_config, f)

    if run:
        from generator.test_engine import TestMode, TestEngine
        from generator.test_engine.test_config import HardwareDebugFileSwitch

        test_phase = []
        # map_config['sim_clock'] = 200_000
        test_config = {
                'tb_name': 'M99999',
                'test_mode': TestMode.MEMORY_STATE,
                'debug_file_switch': HardwareDebugFileSwitch().close_all.singla_chip.dict,
                'test_group_phase': test_phase
            }

        tester = TestEngine(map_config, test_config)
        assert tester.run_test()