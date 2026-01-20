import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G14_Map_Config(phase_en, clock, M, N, start_row=0, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group14 Mapping
    with 32 function cores
    core array : 4 * 8
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
    # 接收数据 10 * 256 + 发送 10 * 32
    if phase_en[0] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1eda0 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=39, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 4
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 5
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 6
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 7
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收数据 10 * 256 + 发送 10 * 32
    if phase_en[1] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1eda0+10*32 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=39, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 4
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 5
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 6
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 7
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+10*256>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收数据 10 * 256 + 发送 10 * 32
    if phase_en[2] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1eda0+20*32 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=39, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 4
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 5
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 6
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 7
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+20*256>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 接收数据 20 * 256 + 发送 19 * 32
    if phase_en[3] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                length = 19 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1eda0+30*32 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=75, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=639, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 4
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 5
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 6
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 7
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+30*256>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 2 **************************************************#
    # 接收数据 50 * 256

    if phase_en[4] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 10699
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=199, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x13200 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 3 **************************************************#
    # 接收数据 48 * 256

    if phase_en[5] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[5], L5_num=delay_L5[5], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 9478
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x16400>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 4 **************************************************#
    # 接收数据 48 * 256
    if phase_en[6] == 1:
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
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 0:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = -1
                    router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19400>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    # ***************************************************** phase 5 **************************************************#
    # L1卷积计算，部分和收发------1 计算奇数行
    if phase_en[7] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=14, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=2, addr_ina=0x10000 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x1c400 >> 2)
            # 只计算奇数行
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[1] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_3_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                        cout=32, px=14, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                        row_ck_on=1, addr_in=0x1c400 >> 2, addr_out=0x9000, in_row_max=7)
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=1567, addr_rhead_base=rhead_base, \
                         addr_rhead_length=1, addr_din_base=0x3c0, addr_din_length=0, receive_num=2, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(1 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y - start_row == 0:
                addr_out = 0x9600 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 1:
                addr_out = 0xa400 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 2:
                addr_out = 0xb200 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 3:
                addr_out = 0xb900 >> 2
                length = 1 * 14 * 32 * 4
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x83c0, addr_ciso=0x0 >> 2,
                        addr_out=addr_out)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 6 **************************************************#
    # L1部分和求和，流水relu，保存在Mem0 0xdc00 ------1 奇数行
    if phase_en[8] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == 0:
                addr_in = 0x8800 >> 2
            elif core_y == 1:
                addr_in = 0x9600 >> 2
            elif core_y == 2:
                addr_in = 0xa400 >> 2
            elif core_y == 3:
                addr_in = 0xb200 >> 2
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, \
                       cin=32, px=2 if core_y != 3 else 1, py=14, kx=2, ky=2, sx=1, sy=1, addr_in=addr_in, \
                       addr_bias=0x0, addr_out=0x1c400 >> 2)
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
                        cout=32, px=2 if core_y != 3 else 1, py=14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=5, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xdc00 >> 2 \
                    , in_row_max=14)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 7 **************************************************#
    # L1卷积计算，部分和收发------2 计算偶数行
    if phase_en[9] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=13, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=2, addr_ina=0x10E00 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
                       addr_out=0x8800 >> 2)
            # 只计算奇数行
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[1] == 0 and phase_en[2] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_5_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )

            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                        cout=32, px=14, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                        row_ck_on=1, addr_in=0x8800 >> 2, addr_out=0x9000, in_row_max=7)
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=1567, addr_rhead_base=rhead_base, \
                         addr_rhead_length=1, addr_din_base=0x3c0, addr_din_length=0, receive_num=2, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(2 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(1 * 14 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0,
                                EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y - start_row == 0:
                addr_out = 0x9600 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 1:
                addr_out = 0xa400 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 2:
                addr_out = 0xb200 >> 2
                length = 2 * 14 * 32 * 4
            elif core_y - start_row == 3:
                addr_out = 0xb900 >> 2
                length = 1 * 14 * 32 * 4
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x83c0, addr_ciso=0x0 >> 2,
                        addr_out=addr_out)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 8 **************************************************#
    # L1部分和求和，流水relu，保存在Mem0 0xea00 ------2 偶数行
    if phase_en[10] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == 0:
                addr_in = 0x8800 >> 2
            elif core_y == 1:
                addr_in = 0x9600 >> 2
            elif core_y == 2:
                addr_in = 0xa400 >> 2
            elif core_y == 3:
                addr_in = 0xb200 >> 2
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, \
                       cin=32, px=2 if core_y != 3 else 1, py=14, kx=2, ky=2, sx=1, sy=1, addr_in=addr_in, \
                       addr_bias=0x0, addr_out=0x1c400 >> 2)
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
                        cout=32, px=2 if core_y != 3 else 1, py=14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=5, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xea00 >> 2 \
                    , in_row_max=14)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=14, py=14, kx=1, ky=1, sx=2, sy=2, cmp_c=0x80808080, \
                        in_cut_start=0, row_ck_on=0, addr_in=0x10000 >> 2, addr_out=0x8800 >> 2 \
                        , in_row_max=3)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 9 **************************************************#
    # L1的结果14 * 14 * 32通过路由整理成14 * 14 * 64 的数据
    if phase_en[11] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y == 3:
                length_in = 448
            else:
                length_in = 896
            soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length_in, length_ciso=length_in, length_out=2*length_in, \
                        num_in=1, num_ciso=1 \
                        , num_out=1 \
                        , row_ck_on=0, addr_in=0xdc00 >> 2,
                        addr_ciso=0xea00 >> 2,
                        type_in=1, type_out=1, addr_out=0x9000)
            if phase_en[7] == 0:
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_7_In".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_7_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0}
                ]
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=0, \
                         addr_rhead_base=rhead_base, addr_rhead_length=1, addr_din_base=0x3c0, addr_din_length=1567, \
                         receive_num=25, t_mode=1, soma_in_en=1)
            if core_x in [0, 1]:
                dst_x = 0
                dst_y = 0
            elif core_x in [2, 3]:
                dst_x = 2
                dst_y = 1
            elif core_x in [4, 5]:
                dst_x = 4
                dst_y = 2
            elif core_x in [6, 7]:
                dst_x = 6
                dst_y = 3
            A = (core_x - dst_x) * 4 + (core_y - start_row) * 448 # 4* 8 阵列
            #A = (core_x % 4) // 2 * 4 + (core_y + (core_x % 2) * 2) * 448
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            if core_y != 3:
                router.Send_number = 223
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=55, \
                                A_offset=4, Const=3, EN=1)# 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A + 224,
                                pack_per_Rhead=55, \
                                A_offset=4, Const=3, EN=1)# 3
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A + 112,
                                pack_per_Rhead=55, \
                                A_offset=4, Const=3, EN=1)# 2
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A + 336,
                                pack_per_Rhead=55, \
                                A_offset=4, Const=3, EN=1)# 4
            else:
                router.Send_number = 111
                router.Addr_Rhead_length = 0
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=111, \
                                A_offset=4, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 10 **************************************************#

    # L1多播，X1数据split，只保留32channel作为Xe
    if phase_en[12] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

            if core_x == 0:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=26, length_in=256, length_ciso=16, length_out=32, \
                            num_in=49, num_ciso=49, num_out=49, row_ck_on=0, addr_in=0x8800 >> 2,
                            addr_ciso=0xdc00 >> 2,
                            type_in=1, type_out=1, addr_out=0x1f9e0 >> 2)
            else:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=26, length_in=256, \
                            length_ciso=32, length_out=32 * core_x, \
                            num_in=49, num_ciso=49, num_out=49, row_ck_on=0, addr_in=0x8800 >> 2,
                            addr_ciso=0x1f9e0 >> 2,
                            type_in=1, type_out=1, addr_out= 0xdc00>> 2)

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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                         addr_dout_base=0x3c0, addr_dout_length=783, send_num=1567, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x3c0, addr_din_length=1567,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if phase_en[8] == 0:
                r = router.init_data()
                router.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_7_router_In".format(core_x, core_y),
                     'start': router.Addr_Dout_base + 0x8000,
                     'length': (router.Addr_Dout_length + 1) * 4,
                     'data': r,
                     'mode': 0}
                ]
            if core_y - start_row == 0:
                if core_x == 0:
                    router.Receive_en = 0
                    router.Send_en = 1
                    router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                    pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
                elif core_x != 7:
                    router.CXY = 1
                    router.Relay_number = 1567
                    router.Nx = 1
                    router.Ny = 0
            elif core_y - start_row == 1:
                if core_x == 2:
                    router.Receive_en = 0
                    router.Send_en = 1
                    router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                    pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
                elif core_x != 7:
                    router.CXY = 1
                    router.Relay_number = 1567
                    if core_x in [1]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 3
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            elif core_y - start_row == 2:
                if core_x == 4:
                    router.Receive_en = 0
                    router.Send_en = 1
                    router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                    pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
                elif core_x != 7:
                    router.CXY = 1
                    router.Relay_number = 1567
                    if core_x in [1, 2, 3]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 5
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            elif core_y - start_row == 3:
                if core_x == 6:
                    router.Receive_en = 0
                    router.Send_en = 1
                    router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                    pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1567
                    if core_x == 7:
                        router.Nx = -2
                        router.Ny = 0
                    else:
                        router.Nx = -1
                        router.Ny = 0

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=64, \
                        cout=64, px=14, py=14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 11 **************************************************#
    # L2卷积计算，部分和收发（进行抽帧）
    if phase_en[13] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=14, pad_on=True, load_bias=0, cin=64, cout=32, \
                       kx=3, ky=3, sx=2, sy=2, addr_ina=0x1b680 >> 2, addr_inb=0x2000 >> 2, addr_bias=0x0, \
                       pad_top=1, pad_down=0, pad_right=1, pad_left=0, addr_out=0x8800 >> 2)
            # 只计算奇数行
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[7] == 0:
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
                        row_ck_on=1, addr_in=0x8800 >> 2, addr_out=0x9000, in_row_max=7)
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=783, addr_rhead_base=rhead_base, \
                         addr_rhead_length=1, addr_din_base=0x3c0, addr_din_length=0, receive_num=2, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(2 * 7 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(2 * 7 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(2 * 7 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=0)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(1 * 7 * 32 * 4 * 3 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y - start_row == 0:
                addr_out = 0x8f00 >> 2
                length = 2 * 7 * 32 * 4
            elif core_y - start_row == 1:
                addr_out = 0x9600 >> 2
                length = 2 * 7 * 32 * 4
            elif core_y - start_row == 2:
                addr_out = 0x9d00 >> 2
                length = 2 * 7 * 32 * 4
            elif core_y - start_row == 3:
                addr_out = 0xa080 >> 2
                length = 1 * 7 * 32 * 4
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x83c0, addr_ciso=0x0 >> 2,
                        addr_out=addr_out)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 12 **************************************************#
    # L2部分和求和，流水relu，保存在Mem0 0xdc00
    if phase_en[14] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y == 0:
                addr_in = 0x8800 >> 2
            elif core_y == 1:
                addr_in = 0x8f00 >> 2
            elif core_y == 2:
                addr_in = 0x9600 >> 2
            elif core_y == 3:
                addr_in = 0x9d00 >> 2
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, \
                       cin=32, px=2 if core_y != 3 else 1, py=7, kx=2, ky=2, sx=1, sy=1, addr_in=addr_in, \
                       addr_bias=0x0, addr_out=0x16200 >> 2)
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

            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=2 if core_y != 3 else 1, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=4, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xdc00 >> 2 \
                    , in_row_max=7)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 13 **************************************************#
    # L2的结果整理 从7*7*32 -> 7*7*512 发送
    if phase_en[15] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y == 3:
                length = 7 * 1 * 32
            else:
                length = 7 * 2 * 32
            soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xdc00 >> 2,
                        addr_ciso=0x0 >> 2,
                        type_in=1, type_out=1, addr_out=0x9000)
            if phase_en[9] == 0:
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_13_In".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                    {'name': "core_{:d}_{:d}_phase_13_ciso".format(core_x, core_y),
                     'start': soma1.Addr_Start_ciso,
                     'length': soma1.length_ciso * soma1.num_ciso // 4,
                     'data': s1[1],
                     'mode': 0}
                ]

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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=length // 8 - 1, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x3c0, addr_din_length=1567, \
                         receive_num=31, t_mode=1, soma_in_en=1)
            dst_x = 0
            dst_y = 0
            # A = core_x // 2 * 4 + (core_y + (core_x % 2) * 2) * 448   # 2 * 16
            A = core_x * 4 + core_y * 448
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, \
                                A_offset=28, Const=3, EN=1)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 14 **************************************************#

    # L2的结果整理 从7*7*32 -> 7*7*512 多播
    if phase_en[16] == 1:
        assert(phase_en[10] == 1)# 需要与上一个phase一起跑
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                         addr_dout_base=0x3c0, addr_dout_length=783, send_num=1567, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x3c0, addr_din_length=1567,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if core_y % 2 == 0:
                if core_x == M - 1 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1567
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != M - 1:
                    router.CXY = 1
                    router.Relay_number = 1567
                    router.Nx = 1
                    router.Ny = 0
            else:
                if core_x == 0 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1567
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1567
                    router.Nx = -1
                    router.Ny = 0
            if core_y== 0 and core_x == 0:
                router.Receive_en = 0
                router.Send_en = 1
                router.CXY = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 15 **************************************************#
    # L3层卷积计算
    if phase_en[17] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=7, py=7, pad_on=False,\
                       load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x6800 >> 2, addr_bias=0x0, \
                       addr_out=0x8800 >> 2)
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_21_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[11] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_21_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, in_cut_start=4, \
                        row_ck_on=1, addr_in=0x8800 >> 2, addr_out=0x1f3c0 >> 2, in_row_max=7)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)


            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 16 **************************************************#
    # L3层与shorcut结果求和，流水relu，保存在Mem1 0x1eda0
    if phase_en[18] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1, load_bias=0, \
                       cin=32, px=7, py=7, kx=2, ky=1, sx=1, sy=1, addr_in=0x1f3c0 >> 2, \
                       addr_bias=0x0, addr_out=0x8800 >> 2)
            if phase_en[12] == 0 or phase_en[7] == 0:
                assert(phase_en[12] == 0 and phase_en[7] == 0)  # 需要单独跑，或者三个phase一起跑
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_22_InA".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky // 2,
                     'data': a[0],
                     'mode': 0}
                ]
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=7, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=1, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1eda0 >> 2, in_row_max=7)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    return map_config


# # ---------------------------------------------------------------------------------------#
# import os
# import sys
# case_file_name = 'M99999'
# c_path = os.getcwd()
# out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
# if os.path.exists(out_files_path):
#     os.chdir(out_files_path)
#     del_command = 'rd/s/q cmp_out'
#     os.system(del_command)
#     os.chdir(c_path)
# # 真正执行的代码
# phase_en = np.ones(32).astype(int)
# # 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
# # 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
# # group间
# phase_en[0] = 0     # 组间
# phase_en[1] = 0     # 组间
# phase_en[2] = 0     # 组间
# phase_en[3] = 0     # 组间
#
# phase_en[4] = 1     # L1卷积计算，部分和收发------1 计算奇数行
# phase_en[5] = 1     # L1部分和求和，流水relu， ------1 奇数行
# phase_en[6] = 1     # L1卷积计算，部分和收发------2 计算偶数行
# phase_en[7] = 1     # L2部分和求和，流水relu， ------1 偶数行
# phase_en[8] = 1     # L1的结果通过路由整理成14 * 14 * 64 的数据
# phase_en[9] = 1     # L1多播，X1数据split，只保留32channel作为Xe
#
# phase_en[10] = 1     # L2卷积计算，部分和收发
# phase_en[11] = 1     # L2部分和求和，流水relu
#
# phase_en[12] = 1    # L2的结果整理 从7*7*32 -> 7*7*25 发送
# phase_en[13] = 1    # L2的结果整理 从7*7*32 -> 7*7*25 多播
#
# phase_en[14] = 1    # L3层卷积计算，流水截取
# phase_en[15] = 1    # L3层与shorcut结果求和，流水relu，保存在Mem1 0x1eda0
#
#
#
# run = True  # 只生成map_config 还是生成后直接运行
#
# map_config = Gen_G14_Map_Config(phase_en, 28000, M=8 , N=4)
#
# from generator.test.Multi_Groups.AddRouterInfo import add_router_info
# map_config = add_router_info(map_config=map_config, group_idx_list=[0], chip_x_num=1, chip_y_num=1, core_x_num=16, core_y_num=10)
#
# #import pickle
# #with open('G16_64cores\\G16_64cores_phase_1_12', 'wb') as f:
# #    pickle.dump(map_config, f)
#
# if run:
#     from generator.test_engine import TestMode, TestEngine
#
#     test_phase = []
#     for i in range(len(map_config[0][0][((0, 0), (0, 0))]['axon'])):
#         test_phase.append((0, i + 1))
#     map_config['sim_clock'] = min(len(map_config[0][0][((0, 0), (0, 0))]['axon']) * map_config[0][0]['clock'] - 1, 160000)
#     test_config = {
#             'tb_name': 'M99999',
#             'test_mode': TestMode.MEMORY_STATE,
#             'test_group_phase': test_phase
#         }
#
#     tester = TestEngine(map_config, test_config)
#     assert tester.run_test()