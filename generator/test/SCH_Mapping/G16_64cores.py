import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G16_Map_Config(phase_en, clock, M, N, start_row=0, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group15 Mapping
    with 64 function cores
    and 4 buffer cores (to simulate sending data from Group 14)
    core array : 5 *16 ( the last row core is the buffer cores)
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
    # 接收5 * 512的数据，并横向多播
    if phase_en[0] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:#最后一行core是数据发送core，最后一行core也要接收Group15计算完成后发来的数据
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x2000, addr_inb=0x2000, addr_bias=0x2000, \
                           addr_out=0x2000, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
                # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                # 11994
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
                if out_data_en == 1:
                    if core_y in [0]:
                        length = 5*128
                        soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                                    length_out=length, \
                                    num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1f200 >> 2,
                                    addr_ciso=0x9000 >> 2,
                                    addr_out=0x9000)
                        s1 = soma1.init_data()
                        soma1.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                             'start': soma1.Addr_Start_in,
                             'length': soma1.length_in * soma1.num_in // 4,
                             'data': s1[0],
                             'mode': 0},
                        ]
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
                    else:
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
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
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                             addr_dout_length=0, send_num=79, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                             addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
                if in_data_en == 1:
                    router.Receive_en = 1
                    if core_y - start_row == 0:
                        if core_x == 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                    elif core_y - start_row == 1:
                        if core_x == 4:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 5
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 2:
                        if core_x == 8:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3, 4, 5, 6, 7]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 9
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 3:
                        if core_x == 12:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [13, 14]:
                                router.Nx = 1
                                router.Ny = 0
                            elif core_x == 15:
                                router.Nx = -4
                                router.Ny = 0
                            elif core_x != 0:
                                router.Nx = -1
                                router.Ny = 0
                if out_data_en == 1:
                    if core_y in [0]:
                        if core_y == 0:
                            router.Send_en = 1
                            router.Soma_in_en = 1
                        if core_x in [0, 1, 2, 3]:
                            dst_x = 0
                            dst_y = 4
                        elif core_x in [4, 5, 6, 7]:
                            dst_x = 4
                            dst_y = 5
                        elif core_x in [8, 9, 10, 11]:
                            dst_x = 8
                            dst_y = 6
                        elif core_x in [12, 13, 14, 15]:
                            dst_x = 12
                            dst_y = 7
                        A = (core_x - dst_x) * 16 + core_y * 896
                        dst_y = dst_y - 8
                        router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                            pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if in_data_en == 1:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                                cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000 >> 2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收5 * 512的数据，并横向多播
    if phase_en[1] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:#最后一行core是数据发送core，最后一行core也要接收Group15计算完成后发来的数据
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x2000, addr_inb=0x2000, addr_bias=0x2000, \
                           addr_out=0x2000, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
                # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                # 11994
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
                if out_data_en == 1:
                    if core_y in [0]:
                        length = 5*128
                        soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                                    length_out=length, \
                                    num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1f200+5*128 >> 2,
                                    addr_ciso=0x9000 >> 2,
                                    addr_out=0x9000)
                        s1 = soma1.init_data()
                        soma1.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                             'start': soma1.Addr_Start_in,
                             'length': soma1.length_in * soma1.num_in // 4,
                             'data': s1[0],
                             'mode': 0},
                        ]
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
                    else:
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
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
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                             addr_dout_length=0, send_num=79, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                             addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
                if in_data_en == 1:
                    router.Receive_en = 1
                    if core_y - start_row == 0:
                        if core_x == 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                    elif core_y - start_row == 1:
                        if core_x == 4:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 5
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 2:
                        if core_x == 8:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3, 4, 5, 6, 7]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 9
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 3:
                        if core_x == 12:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [13, 14]:
                                router.Nx = 1
                                router.Ny = 0
                            elif core_x == 15:
                                router.Nx = -4
                                router.Ny = 0
                            elif core_x != 0:
                                router.Nx = -1
                                router.Ny = 0
                if out_data_en == 1:
                    if core_y in [0]:
                        if core_y == 0:
                            router.Send_en = 1
                            router.Soma_in_en = 1
                        if core_x in [0, 1, 2, 3]:
                            dst_x = 0
                            dst_y = 4
                        elif core_x in [4, 5, 6, 7]:
                            dst_x = 4
                            dst_y = 5
                        elif core_x in [8, 9, 10, 11]:
                            dst_x = 8
                            dst_y = 6
                        elif core_x in [12, 13, 14, 15]:
                            dst_x = 12
                            dst_y = 7
                        A = (core_x - dst_x) * 16 + core_y * 896
                        dst_y = dst_y - 8
                        router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                            pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if in_data_en == 1:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                                cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+5*512 >> 2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收5 * 512的数据，并横向多播
    if phase_en[2] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:#最后一行core是数据发送core，最后一行core也要接收Group15计算完成后发来的数据
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x2000, addr_inb=0x2000, addr_bias=0x2000, \
                           addr_out=0x2000, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
                # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                # 11994
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
                if out_data_en == 1:
                    if core_y in [0, 1]:
                        if core_y == 0:
                            length = 4*128
                            addr_in = 0x1f200 + 10*128 >> 2
                        else:
                            length = 1*128
                            addr_in = 0x1f200 >> 2
                        soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                                    length_out=length, \
                                    num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                                    addr_ciso=0x9000 >> 2,
                                    addr_out=0x9000)
                        s1 = soma1.init_data()
                        soma1.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                             'start': soma1.Addr_Start_in,
                             'length': soma1.length_in * soma1.num_in // 4,
                             'data': s1[0],
                             'mode': 0},
                        ]
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
                    else:
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
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
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                             addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                             addr_din_base=0x380, addr_din_length=319, receive_num=0, t_mode=1, soma_in_en=0)
                if in_data_en == 1:
                    router.Receive_en = 1
                    if core_y - start_row == 0:
                        if core_x == 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                    elif core_y - start_row == 1:
                        if core_x == 4:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 5
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 2:
                        if core_x == 8:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3, 4, 5, 6, 7]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 9
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 3:
                        if core_x == 12:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [13, 14]:
                                router.Nx = 1
                                router.Ny = 0
                            elif core_x == 15:
                                router.Nx = -4
                                router.Ny = 0
                            elif core_x != 0:
                                router.Nx = -1
                                router.Ny = 0
                if out_data_en == 1:
                    if core_y in [0, 1]:
                        router.Send_en = 1
                        router.Soma_in_en = 1
                        if core_y == 0:
                            router.Send_number = 4*128//8-1
                        else:
                            router.Send_number = 1*128//8-1
                        if core_x in [0, 1, 2, 3]:
                            dst_x = 0
                            dst_y = 4
                        elif core_x in [4, 5, 6, 7]:
                            dst_x = 4
                            dst_y = 5
                        elif core_x in [8, 9, 10, 11]:
                            dst_x = 8
                            dst_y = 6
                        elif core_x in [12, 13, 14, 15]:
                            dst_x = 12
                            dst_y = 7
                        A = (core_x - dst_x) * 16 + core_y * 256
                        dst_y = dst_y - 8
                        router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                            pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if in_data_en == 1:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                                cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+10*512 >> 2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收10 * 512的数据，并横向多播
    if phase_en[3] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:#最后一行core是数据发送core，最后一行core也要接收Group15计算完成后发来的数据
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x2000, addr_inb=0x2000, addr_bias=0x2000, \
                           addr_out=0x2000, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
                # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
                # 11994
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
                if out_data_en == 1:
                    if core_y in [1]:
                        length = 10*128
                        soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                                    length_out=length, \
                                    num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1f200+128 >> 2,
                                    addr_ciso=0x9000 >> 2,
                                    addr_out=0x9000)
                        s1 = soma1.init_data()
                        soma1.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                             'start': soma1.Addr_Start_in,
                             'length': soma1.length_in * soma1.num_in // 4,
                             'data': s1[0],
                             'mode': 0},
                        ]
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
                    else:
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
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
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                             addr_dout_length=0, send_num=159, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                             addr_din_base=0x380, addr_din_length=639, receive_num=0, t_mode=1, soma_in_en=0)
                if in_data_en == 1:
                    router.Receive_en = 1
                    if core_y - start_row == 0:
                        if core_x == 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                    elif core_y - start_row == 1:
                        if core_x == 4:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 5
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 2:
                        if core_x == 8:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [1, 2, 3, 4, 5, 6, 7]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 9
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 3:
                        if core_x == 12:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != 0:
                            router.CXY = 1
                            router.Relay_number = router.Addr_Din_length
                            if core_x in [13, 14]:
                                router.Nx = 1
                                router.Ny = 0
                            elif core_x == 15:
                                router.Nx = -4
                                router.Ny = 0
                            elif core_x != 0:
                                router.Nx = -1
                                router.Ny = 0
                if out_data_en == 1:
                    if core_y in [1]:
                        if core_y == 1:
                            router.Send_en = 1
                            router.Soma_in_en = 1
                        if core_x in [0, 1, 2, 3]:
                            dst_x = 0
                            dst_y = 4
                        elif core_x in [4, 5, 6, 7]:
                            dst_x = 4
                            dst_y = 5
                        elif core_x in [8, 9, 10, 11]:
                            dst_x = 8
                            dst_y = 6
                        elif core_x in [12, 13, 14, 15]:
                            dst_x = 12
                            dst_y = 7
                        A = (core_x - dst_x) * 16
                        dst_y = dst_y - 8
                        router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                            pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if in_data_en == 1:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                                cout=512, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+15*512 >> 2)
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 2 **************************************************#
    """
        接收24 * 512的数据，并横向多播
    """
    if phase_en[4] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是数据发送core，最后一行core也要接收Group15计算完成后发来的数据
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                if out_data_en == 1:
                    if core_y in [1, 2, 3]:
                        if core_y == 1:
                            length = 3 * 128
                            addr_in = (0x1f200 + 4 * 128) >> 2
                        elif core_y == 2:
                            length = 7 * 2 * 128
                            addr_in = 0x1f200 >> 2
                        elif core_y == 3:
                            length = 7 * 1 * 128
                            addr_in = 0x1f200 >> 2
                        soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                                    length_out=length, \
                                    num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in,
                                    addr_ciso=0x9000 >> 2,
                                    addr_out=0x9000)
                        s1 = soma1.init_data()
                        soma1.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                             'start': soma1.Addr_Start_in,
                             'length': soma1.length_in * soma1.num_in // 4,
                             'data': s1[0],
                             'mode': 0},
                        ]
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
                    else:
                        map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
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
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                             addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                             addr_din_base=0x380, addr_din_length=1535, receive_num=0, t_mode=1, soma_in_en=1)
                if in_data_en == 1:
                    router.Receive_en = 1
                    if core_y - start_row == 0:
                        if core_x == 0:
                            router.CXY = 1
                            router.Relay_number = 1535
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = 1535
                            router.Nx = 1
                            router.Ny = 0
                    elif core_y - start_row == 1:
                        if core_x == 4:
                            router.CXY = 1
                            router.Relay_number = 1535
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = 1535
                            if core_x in [1, 2, 3]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 5
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 2:
                        if core_x == 8:
                            router.CXY = 1
                            router.Relay_number = 1535
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.CXY = 1
                            router.Relay_number = 1535
                            if core_x in [1, 2, 3, 4, 5, 6, 7]:
                                router.Nx = -1
                                router.Ny = 0
                            elif core_x == 0:
                                router.Nx = 9
                                router.Ny = 0
                            elif core_x != M - 1:
                                router.Nx = 1
                                router.Ny = 0
                    elif core_y - start_row == 3:
                        if core_x == 12:
                            router.CXY = 1
                            router.Relay_number = 1535
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x != 0:
                            router.CXY = 1
                            router.Relay_number = 1535
                            if core_x in [13, 14]:
                                router.Nx = 1
                                router.Ny = 0
                            elif core_x == 15:
                                router.Nx = -4
                                router.Ny = 0
                            elif core_x != 0:
                                router.Nx = -1
                                router.Ny = 0
                if out_data_en == 1:
                    if core_y in [1, 2, 3]:
                        if core_x in [0, 1, 2, 3]:
                            dst_x = 0
                            dst_y = 4   # -4
                        elif core_x in [4, 5, 6, 7]:
                            dst_x = 4
                            dst_y = 5   # -3
                        elif core_x in [8, 9, 10, 11]:
                            dst_x = 8
                            dst_y = 6   # -2
                        elif core_x in [12, 13, 14, 15]:
                            dst_x = 12
                            dst_y = 7   # -1
                        if core_y == 1:
                            router.Send_en = 1
                            router.Send_number = 3 * 128 // 8 - 1
                            router.Soma_in_en = 1
                            A = (core_x - dst_x) * 16
                        elif core_y == 2:
                            router.Send_en = 1
                            router.Send_number = 7 * 2 * 128 // 8 - 1
                            router.Soma_in_en = 1
                            A = (core_x - dst_x) * 16 + 192
                        elif core_y == 3:
                            router.Send_en = 1
                            router.Send_number = 7 * 128 // 8 - 1
                            router.Soma_in_en = 1
                            A = (core_x - dst_x) * 16 + 192 + 896
                        dst_y = dst_y - 8
                        router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                        pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if in_data_en == 1:
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                                cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x1c200 >> 2)
                else:
                    soma2 = None
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 3 **************************************************#
    """
        44卷积计算，部分和收发
    """
    if phase_en[5] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=7, pad_on=False, load_bias=0, cin=512, cout=32, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2, addr_inb=0x0, addr_bias=0x0, \
                           addr_out=0x8000 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_3_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[1] == 0 or in_data_en == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_3_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )

                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x8000 >> 2, addr_out=0x9000, in_row_max=7)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                temp_phase = phase
                rhead_base = 0x300
                if temp_phase > 0:
                    while temp_phase > 0:
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                            break
                        else:
                            temp_phase -= 1
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1] != None:
                        rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_base
                        if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                            rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Addr_Rhead_length + 1) * 4
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=1, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=783, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y - start_row == 0:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 1:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 2:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 3:
                    router.Addr_Din_length = int(1 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4 if core_y - start_row == 3 else 8, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x8000 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
    # ***************************************************** phase 4 **************************************************#
    """
        44部分和求和，为45层卷积计算传输数据
    """
    if phase_en[6] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0,\
                           cin=32, px=1, py=7 if core_y - start_row == 3 else 14, kx=2, ky=2, sx=1, sy=1, addr_in=0x8000>>2, \
                           addr_bias=0x0, addr_out=0x1f200>>2)
                if phase_en[2] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_4_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                            cout=32, px=1, py=7 if core_y == 3 else 14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0,\
                            in_cut_start=2, row_ck_on=1, addr_in=0x1f200 >> 2, addr_out=0x9000, in_row_max=7 if core_y\
                            == 3 else 14)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=0, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=27 if core_y == 3 else 55, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=783,\
                             receive_num=15, t_mode=1, soma_in_en=1)
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_y = start_row
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_y = 1 + start_row
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_y = 2 + start_row
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_y = 3 + start_row
                if core_y == 3:
                    pack_per_rhead = 27
                    A = (core_x - dst_x) * 4 + (core_y - start_row) * 224
                else:
                    pack_per_rhead = 55
                    A = (core_x - dst_x) * 4 + (core_y - start_row) * 224
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x-core_x, Y=dst_y-core_y, A=A, pack_per_Rhead=pack_per_rhead,\
                                A_offset=12, Const=3, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    # ***************************************************** phase 25 **************************************************#
    """
        44多播
    """
    if phase_en[7] == 1:
        assert(phase_en[3] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                             addr_dout_base=0x400, addr_dout_length=391, send_num=0, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=783,\
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y - start_row == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        router.Nx = 1
                        router.Ny = 0
                elif core_y - start_row == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y - start_row == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y - start_row == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [13, 14]:
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x == 15:
                            router.Nx = -4
                            router.Ny = 0
                        elif core_x != 0:
                            router.Nx = -1
                            router.Ny = 0

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=32, \
                            cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400 >> 2, addr_out=0x8000>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 5 **************************************************#
    """
        从X1数据中截取，只保留128channel数据作为Xe
    """
    if phase_en[8] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)


                soma1 = pX6(core_x=core_x, core_y=core_y, type=26, length_in=512, length_ciso=32, length_out=128, \
                            num_in=49, num_ciso=49, num_out=49, row_ck_on=0, addr_in=0x19000 >> 2,
                            addr_ciso=0xa000 >> 2,
                            type_in=1, type_out=1, addr_out=0x9000)
                if phase_en[0] == 0 or phase_en[1] == 0 or in_data_en == 0:
                    s1 = soma1.init_data()
                    soma1.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma1_in".format(core_x, core_y),
                         'start': soma1.Addr_Start_in,
                         'length': soma1.length_in * soma1.num_in // 4,
                         'data': s1[0],
                         'mode': 0}
                    ]
                if core_x // 4 == core_y - start_row:
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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=783, addr_rhead_base=rhead_base, \
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=223, receive_num=0, \
                             t_mode=1, soma_in_en=1)
                if core_x // 4 == core_y - start_row:
                    router.Send_en = 1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0-core_y + start_row, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1-core_y + start_row, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2-core_y + start_row, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3-core_y + start_row, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                if core_y == 3:
                    router.Addr_Din_length = 111
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16, length_out=128, \
                            num_in=14 if core_y == 0 else 7, num_ciso=14 if core_y == 0 else 7\
                            , num_out=14 if core_y == 0 else 7\
                            , row_ck_on=0, addr_in=0x8400,
                            addr_ciso=0xa000 >> 2,
                            type_in=1, type_out=1, addr_out=0x1f900 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 6 **************************************************#
    """
        45层卷积计算
    """
    if phase_en[9] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=7, pad_on=True, pad_top=1, pad_down=1, pad_right=1,\
                           pad_left=1, load_bias=0, cin=128, cout=32, \
                           kx=3, ky=3, sx=1, sy=1, addr_ina=0x8000 >> 2, addr_inb=0x4000, addr_bias=0x0, \
                           addr_out=0x19000 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_6_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[24] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_6_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x19000 >> 2, addr_out=0x9000, in_row_max=7)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=1, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=783, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y - start_row== 0:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 1:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 2:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 3:
                    router.Addr_Din_length = int(1 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4 if core_y - start_row == 3 else 8, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x19000 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 7 **************************************************#
    """
        45部分和求和，为46层卷积计算传输数据
    """
    if phase_en[10] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0, \
                           cin=32, px=1, py=7 if core_y - start_row == 3 else 14, kx=2, ky=2, sx=1, sy=1, addr_in=0x19000 >> 2, \
                           addr_bias=0x0, addr_out=0x8000 >> 2)
                if phase_en[5] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_4_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                            cout=32, px=1, py=7 if core_y - start_row == 3 else 14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=2, row_ck_on=1, addr_in=0x8000 >> 2, addr_out=0x9000, in_row_max=7 if core_y \
                            - start_row == 3 else 14)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=0, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=27 if core_y == 3 else 55, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=783, \
                             receive_num=15, t_mode=1, soma_in_en=1)
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_y = start_row
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_y = 1 + start_row
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_y = 2 + start_row
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_y = 3 + start_row
                if core_y == 3:
                    pack_per_rhead = 27
                    A = (core_x - dst_x) * 4 + (core_y - start_row) * 224
                else:
                    pack_per_rhead = 55
                    A = (core_x - dst_x) * 4 + (core_y - start_row) * 224
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=pack_per_rhead, \
                                A_offset=12, Const=3, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 26 **************************************************#
    """
        45层多播
    """
    if phase_en[11] == 1:
        assert(phase_en[6] == 1)
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                             addr_dout_base=0x400, addr_dout_length=391, send_num=0, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=783, \
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y - start_row == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        router.Nx = 1
                        router.Ny = 0
                elif core_y - start_row == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y - start_row == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y - start_row == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 783
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 783
                        if core_x in [13, 14]:
                            router.Nx = 1
                            router.Ny = 0
                        elif core_x == 15:
                            router.Nx = -4
                            router.Ny = 0
                        elif core_x != 0:
                            router.Nx = -1
                            router.Ny = 0

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=32, \
                            cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400 >> 2, addr_out=0x19000 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 9 **************************************************#
    """
        46层卷积计算，部分和收发---1
    """
    if phase_en[12] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=7, pad_on=False, load_bias=0, cin=128, cout=128, \
                           kx=1, ky=1, sx=1, sy=2, addr_ina=0x19000 >> 2, addr_inb=0x4000 >> 2, addr_bias=0x0, \
                           addr_out=0x8000 >> 2)
                # 只计算奇数行
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[25] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_9_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )

                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x8000 >> 2, addr_out=0x9000, in_row_max=4)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=1, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=1791, addr_rhead_base=rhead_base, \
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=2, \
                             t_mode=1, soma_in_en=1)
                if core_y - start_row == 0:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 1:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 2:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 3:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1344, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if core_y - start_row == 0:
                    addr_out = 0x8e00 >> 2
                elif core_y - start_row == 1:
                    addr_out = 0x9c00 >> 2
                elif core_y - start_row == 2:
                    addr_out = 0xaa00 >> 2
                elif core_y - start_row == 3:
                    addr_out = 0x8000 >> 2
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=3584, length_ciso=16, length_out=3584, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x8400, addr_ciso=0xe200 >> 2,
                            addr_out=addr_out)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 10 **************************************************#
    """
        46层部分和求和，流水截取 ---- 1
    """
    if phase_en[13] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0 or core_y == 3:
                    addr_in = 0x8000 >> 2
                elif core_y == 1:
                    addr_in = 0x8e00 >> 2
                elif core_y == 2:
                    addr_in = 0x9c00 >> 2
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                           load_bias=0, \
                           cin=128, px=1, py=7, kx=2, ky=2, sx=1, sy=1, addr_in=addr_in, \
                           addr_bias=0x0, addr_out=0xe200 >> 2)
                if phase_en[8] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_4_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=1, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=2, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1f200 >> 2 \
                        if core_y != 3 else 0x1f580 >> 2, in_row_max=7)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 11 **************************************************#
    """
        46层卷积计算，部分和收发---2
    """
    if phase_en[14] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=5, pad_on=False, load_bias=0, cin=128, cout=128, \
                           kx=1, ky=1, sx=1, sy=2, addr_ina=0x19380 >> 2, addr_inb=0x4000 >> 2, addr_bias=0x0, \
                           addr_out=0x8000 >> 2)
                # 只计算偶数行
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_11_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[25] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_11_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )

                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x8000 >> 2, addr_out=0x9000, in_row_max=3)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

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
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=1, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=1343, addr_rhead_base=rhead_base, \
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=2, \
                             t_mode=1, soma_in_en=1)
                if core_y - start_row == 0:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 1:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                elif core_y - start_row == 2:
                    router.Addr_Din_length = int(1 * 7 * 128 * 4 * 3 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                elif core_y - start_row == 3:
                    router.Receive_en = 0
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                if core_y - start_row == 0:
                    addr_out = 0x8e00 >> 2
                elif core_y - start_row == 1:
                    addr_out = 0x9c00 >> 2
                elif core_y - start_row == 2:
                    addr_out = 0x8000 >> 2
                elif core_y - start_row == 3:
                    addr_out = 0x0 >> 2#不需要soma2
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=3584, length_ciso=16, length_out=3584, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x8400, addr_ciso=0xe200 >> 2,
                            addr_out=addr_out)
                if core_y - start_row == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 12 **************************************************#
    """
        46层部分和求和，流水截取 ----2
    """
    if phase_en[15] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y - start_row == 0 or core_y - start_row == 2:
                    addr_in = 0x8000 >> 2
                elif core_y - start_row == 1:
                    addr_in = 0x8e00 >> 2
                elif core_y - start_row == 3:
                    addr_in = 0x0 >> 2#不需要axon
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                           load_bias=0, \
                           cin=128, px=1, py=7, kx=2, ky=2, sx=1, sy=1, addr_in=addr_in, \
                           addr_bias=0x0, addr_out=0xe200 >> 2)
                if phase_en[10] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_12_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                if core_y - start_row == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=1, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=2, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1f580 >> 2, in_row_max=7)
                if core_y - start_row == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 13 **************************************************#
    """
        46层与shorcut结果求和，流水relu，保存在mem1 0x1f200
    """
    if phase_en[16] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y - start_row == 3:
                    px = 1
                    py = 7
                    addr_in = 0x1f900 >> 2
                else:
                    px = 2
                    py = 7
                    addr_in = 0x1f200 >> 2
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1, load_bias=0, \
                           cin=128, px=px, py=py, kx=2, ky=1, sx=1, sy=1, addr_in=addr_in, \
                           addr_bias=0x0, addr_out=0x8000 >> 2)
                if phase_en[9] == 0 or phase_en[11] == 0 or phase_en[4] == 0:
                    assert(phase_en[9] == 0 and phase_en[11] == 0 and phase_en[4] == 0)  # 需要单独跑，或者三个phase一起跑
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_13_InA_1".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=1 if core_y-start_row==3 else 2, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=2, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1f200 >> 2, in_row_max=7)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    return map_config


"""# ---------------------------------------------------------------------------------------#
# 真正执行的代码
phase_en = np.zeros(32)
# 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
# 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
# group间
phase_en[0] = 0     # 组间
phase_en[1] = 0     # 组间

phase_en[2] = 1     # 44卷积计算，部分和收发
phase_en[3] = 1     # 44部分和求和
phase_en[24] = 1    # 44多播
phase_en[4] = 1     # X1数据整理成Xe

phase_en[5] = 1     # 45层卷积计算   --- 时钟较大 2.2k可以满足
phase_en[6] = 1     # 45部分和求和
phase_en[25] = 1    # 45层多播

phase_en[8] = 1     # 46层卷积，部分和收发 ---1
phase_en[9] = 1     # 46层部分和求和，流水截取 ---1
phase_en[10] = 1    # 46层卷积，部分和收发 ---2
phase_en[11] = 1    # 46层部分和求和，流水截取 ---2
phase_en[12] = 1    # 46层输出与Xe相加，输出保存在Mem1 0x19000

run = True  # 只生成map_config 还是生成后直接运行

map_config = Gen_G16_Map_Config(phase_en, 22000, M=16, N=5)

import pickle
with open('G16_64cores\\G16_64cores_phase_1_12', 'wb') as f:
    pickle.dump(map_config, f)

if run:
    from generator.test_engine import TestMode, TestEngine

    test_phase = []
    for i in range(len(map_config[0][0][((0, 0), (0, 0))]['axon'])):
        test_phase.append((0, i + 1))
    map_config['sim_clock'] = min(len(map_config[0][0][((0, 0), (0, 0))]['axon']) * map_config[0][0]['clock'] - 1, 100000)
    test_config = {
            'tb_name': 'M99999',
            'test_mode': TestMode.MEMORY_STATE,
            'test_group_phase': test_phase
        }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()"""