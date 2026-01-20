import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G11_Map_Config(phase_en, clock, M, N, start_row=0, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group11 Mapping
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
    # 接收数据 + 发送 10 * 256
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
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1c400 >> 2,
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
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
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
    # 接收数据 + 发送 10 * 256
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
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1c400+10*32 >> 2,
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
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
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
    # 接收数据 + 发送 10 * 256
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
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 10 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1c400+20*32 >> 2,
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
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
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
    # 接收数据 + 发送 20 * 256
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
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 20 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1c400+30*32 >> 2,
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
                         addr_dout_length=0, send_num=79, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=639, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
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
    # 接收数据 + 发送 50 * 256

    if phase_en[4] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 50 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=(0x1c400 + 50 * 32) >> 2,
                            addr_ciso=0x0 >> 2,
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
                         addr_dout_length=0, send_num=199, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x13200>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 3 **************************************************#
    # 接收数据 + 发送 48 * 256

    if phase_en[5] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x8800 >> 2, addr_inb=0x8800 >> 2, addr_bias=0x8800 >> 2, \
                       addr_out=0x8800 >> 2, axon_delay=True, L4_num=delay_L4[5], L5_num=delay_L5[5], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 48 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=(0x1c400 + 2 * 50 * 32) >> 2,
                            addr_ciso=0x0 >> 2,
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
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x16400>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 4 **************************************************#
    # 接收数据 + 发送 48 * 256
    if phase_en[6] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if out_data_en == 1:
                # 一行8个core发送给下一个group的一行
                length = 48 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16,
                            length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=(0x1c400 + 2 * 50 * 32 + 48 * 32) >> 2,
                            addr_ciso=0x0 >> 2,
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
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=7, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_x != 7:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    router.Nx = 1
                    router.Ny = 0
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = 7
                    dst_y = -4
                elif core_y == 1:
                    dst_x = 6
                    dst_y = -3
                elif core_y == 2:
                    dst_x = 5
                    dst_y = -2
                elif core_y == 3:
                    dst_x = 4
                    dst_y = -1
                A = core_x * 4
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, A_offset=28, Const=3, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19400>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 5 **************************************************#
    # L1卷积计算，部分和收发------1
    if phase_en[7] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=7, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
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
            if phase_en[3] == 0:
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
                         addr_rhead_length=1, addr_din_base=0x380, addr_din_length=0, receive_num=3, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y in [0, 1]:
                addr_out = 0x8800 >> 2
                length = 25 * 32
            else:
                addr_out = 0x8800 >> 2
                length = 24 * 32
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x8380, addr_ciso=0x0 >> 2,
                        addr_out=addr_out, type_in=0, type_out=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 6 **************************************************#
    # L1部分和求和，流水relu，保存在Mem0 0xba00 ------1
    if phase_en[8] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y in [0, 1]:
                py = 25
            else:
                py = 24
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, cin=32, px=1, py=py, kx=2, ky=2, sx=1, sy=1, addr_in=0x8800 >> 2, \
                       addr_bias=0x0, addr_out=0x1c400 >> 2)
            if phase_en[4] == 0:
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
                        cout=32, px=1, py=py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=5, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xba00 >> 2 \
                    , in_row_max=py)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 7 **************************************************#
    # L1卷积计算，部分和收发------2
    if phase_en[9] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=7, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x16200 >> 2, addr_inb=0x0 >> 2, addr_bias=0x0, \
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
            if phase_en[3] == 0:
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
                         addr_rhead_length=1, addr_din_base=0x380, addr_din_length=0, receive_num=3, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y in [0, 1]:
                addr_out = 0x8800 >> 2
                length = 25 * 32
            else:
                addr_out = 0x8800 >> 2
                length = 24 * 32
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x8380, addr_ciso=0x0 >> 2,
                        addr_out=addr_out, type_in=0, type_out=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 8 **************************************************#
    # L1部分和求和，流水relu，保存在Mem0 0xbd20 ------2
    if phase_en[10] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y in [0, 1]:
                py = 25
            else:
                py = 24
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, cin=32, px=1, py=py, kx=2, ky=2, sx=1, sy=1, addr_in=0x8800 >> 2, \
                       addr_bias=0x0, addr_out=0x1c400 >> 2)
            if phase_en[6] == 0:
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
                        cout=32, px=1, py=py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=5, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xbd20 >> 2 \
                    , in_row_max=py)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 9 **************************************************#
    # L1的结果14 * 14 * 32通过路由整理成14 * 14 * 64 的数据
    if phase_en[11] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y in [0, 1]:
                length_in = 25 * 32
            else:
                length_in = 24 * 32
            soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length_in, length_ciso=length_in, length_out=2*length_in, \
                        num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xba00 >> 2,
                        addr_ciso=0xbd20 >> 2, type_in=1, type_out=1, addr_out=0x9000)
            if phase_en[5] == 0 or phase_en[7] == 0:
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
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1567, \
                         receive_num=15, t_mode=1, soma_in_en=1)
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
            # A = (core_x - dst_x) * 56 + (core_y - start_row) * 448 # 4* 8 阵列
            # virtual_core_x = core_x // 2
            # virtual_core_y = core_y + core_x % 2 * 2
            virtual_core_x = core_x
            virtual_core_y = core_y
            if virtual_core_y < 2:
                A = virtual_core_x % 2 * 4 + virtual_core_y * 200
            else:
                A = virtual_core_x % 2 * 4 + 400 + (virtual_core_y - 2) * 192
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            if virtual_core_y < 2:
                router.Send_number = 25 * 32 * 2 // 8 - 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=25 * 32 // 8 - 1, \
                                A_offset=4, Const=3, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A + 784,
                                pack_per_Rhead=25 * 32 // 8 - 1, \
                                A_offset=4, Const=3, EN=1)
            else:
                router.Send_number = 24 * 32 * 2 // 8 - 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=24 * 32 // 8 - 1, \
                                A_offset=4, Const=3, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A + 784,
                                pack_per_Rhead=24 * 32 // 8 - 1, \
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

            soma1 = pX6(core_x=core_x, core_y=core_y, type=26, length_in=256, length_ciso=16, length_out=32, \
                        num_in=49, num_ciso=49, num_out=49, row_ck_on=0, addr_in=(0x10000+core_x//2*32) >> 2,
                        addr_ciso=0x8800 >> 2,
                        type_in=1, type_out=1, addr_out=0xe780 >> 2)
            if phase_en[3] == 0:
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_10_soma1_In".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=1, cxy=0, \
                         addr_dout_base=0x380, addr_dout_length=783, send_num=1567, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1567,\
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
                        in_cut_start=0, row_ck_on=0, addr_in=0x38==8380, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 11 **************************************************#
    # L2卷积计算，部分和收发------1
    if phase_en[13] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=8, pad_on=True, load_bias=0, cin=64, cout=32, \
                       kx=3, ky=3, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x2000 >> 2, addr_bias=0x0, \
                       addr_out=0x13100 >> 2, pad_top=1, pad_down=0, pad_left=1, pad_right=1)
            # 只计算奇数行
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[9] == 0:
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
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x9000, in_row_max=7)
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
                         addr_rhead_length=1, addr_din_base=0x380, addr_din_length=0, receive_num=3, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y in [0, 1]:
                addr_out = 0x8800 >> 2
                length = 25 * 32
            else:
                addr_out = 0x8800 >> 2
                length = 24 * 32
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x8380, addr_ciso=0x0 >> 2,
                        addr_out=addr_out, type_in=0, type_out=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 12 **************************************************#
    # L2部分和求和，流水relu，保存在Mem0 0xba00 ------1
    if phase_en[14] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y in [0, 1]:
                py = 25
            else:
                py = 24
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, cin=32, px=1, py=py, kx=2, ky=2, sx=1, sy=1, addr_in=0x8800 >> 2, \
                       addr_bias=0x0, addr_out=0x13100 >> 2)
            if phase_en[10] == 0:
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
                        cout=32, px=1, py=py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=4, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xba00 >> 2 \
                    , in_row_max=py)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 13 **************************************************#
    # L2卷积计算，部分和收发------2
    if phase_en[15] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=14, py=8, pad_on=True, load_bias=0, cin=64, cout=32, \
                       kx=3, ky=3, sx=1, sy=1, addr_ina=0x11500 >> 2, addr_inb=0x2000 >> 2, addr_bias=0x0, \
                       addr_out=0x13100 >> 2, pad_top=0, pad_down=1, pad_left=1, pad_right=1)
            # 只计算奇数行
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_9_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[9] == 0:
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
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x9000, in_row_max=7)
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
                         addr_rhead_length=1, addr_din_base=0x380, addr_din_length=0, receive_num=3, \
                         t_mode=1, soma_in_en=1)
            if core_y - start_row == 0:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 1:
                router.Addr_Din_length = int(25 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 2:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y - start_row == 3:
                router.Addr_Din_length = int(24 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if core_y in [0, 1]:
                addr_out = 0x8800 >> 2
                length = 25 * 32
            else:
                addr_out = 0x8800 >> 2
                length = 24 * 32
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                        num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x8380, addr_ciso=0x0 >> 2,
                        addr_out=addr_out, type_in=0, type_out=0)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 14 **************************************************#
    # L2部分和求和，流水relu，保存在Mem0 0xbd20 ------2
    if phase_en[16] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            if core_y in [0, 1]:
                py = 25
            else:
                py = 24
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                       load_bias=0, cin=32, px=1, py=py, kx=2, ky=2, sx=1, sy=1, addr_in=0x8800 >> 2, \
                       addr_bias=0x0, addr_out=0x13100 >> 2)
            if phase_en[12] == 0:
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
                        cout=32, px=1, py=py, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=4, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xbd20 >> 2 \
                    , in_row_max=py)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 15 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 1 (50 * 256)
    if phase_en[17] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y in [0, 1]:
                length = 25 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xba00 >> 2,
                            addr_ciso=0x0 >> 2,
                            type_in=1, type_out=1, addr_out=0x9000)
                if phase_en[11] == 0:
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
            else:
                soma1 = None
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=99, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1599, \
                         receive_num=15, t_mode=1, soma_in_en=1)
            dst_x = 0
            dst_y = 0
            # virtual_core_x = core_x // 2
            # virtual_core_y = core_y + core_x % 2 * 2
            virtual_core_x = core_x
            virtual_core_y = core_y
            A = virtual_core_x * 4 + virtual_core_y * 800
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, \
                                A_offset=28, Const=3, EN=1)
            if core_y in [0, 1]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 16 **************************************************#

    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 1 (50 * 256)
    if phase_en[18] == 1:
        assert(phase_en[14] == 1)# 需要与上一个phase一起跑
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
                         addr_dout_base=0x380, addr_dout_length=799, send_num=1599, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1599,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if core_y % 2 == 0:
                if core_x == M - 1 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != M - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 1
                    router.Ny = 0
            else:
                if core_x == 0 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = -1
                    router.Ny = 0
            if core_y== 0 and core_x == 0:
                router.Receive_en = 0
                router.Send_en = 1
                router.CXY = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1599, A_offset=0, Const=0, EN=1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=5, py=10, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 17 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 2 (48 * 256)
    if phase_en[19] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y in [2, 3]:
                length = 24 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xba00 >> 2,
                            addr_ciso=0x0 >> 2,
                            type_in=1, type_out=1, addr_out=0x9000)
                if phase_en[11] == 0:
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
            else:
                soma1 = None
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=95, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1535, \
                         receive_num=15, t_mode=1, soma_in_en=1)
            dst_x = 0
            dst_y = 0
            # virtual_core_x = core_x // 2
            # virtual_core_y = core_y + core_x % 2 * 2
            virtual_core_x = core_x
            virtual_core_y = core_y
            A = virtual_core_x * 4 + virtual_core_y * 768
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, \
                                A_offset=28, Const=3, EN=1)
            if core_y in [2, 3]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Send_en = 0
                    map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 18 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 2 (48 * 256)  L3卷积 - 1 (50 * 256)
    if phase_en[20] == 1:
        #assert(phase_en[16] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=10, py=5, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x6800 >> 2, addr_bias=0x0, \
                       addr_out=0x13200 >> 2)
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_18_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[15] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_18_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=10, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, in_cut_start=4, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xcf00 >> 2, in_row_max=5)
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
                         addr_dout_base=0x380, addr_dout_length=767, send_num=1535, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1535,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if core_y % 2 == 0:
                if core_x == M - 1 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != M - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 1
                    router.Ny = 0
            else:
                if core_x == 0 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = -1
                    router.Ny = 0
            if core_y== 0 and core_x == 0:
                router.Receive_en = 0
                router.Send_en = 1
                router.CXY = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 19 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 3 (50 * 256)
    if phase_en[21] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y in [0, 1]:
                length = 25 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xbd20 >> 2,
                            addr_ciso=0x0 >> 2,
                            type_in=1, type_out=1, addr_out=0x9000)
                if phase_en[11] == 0:
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
            else:
                soma1 = None
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=99, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1599, \
                         receive_num=15, t_mode=1, soma_in_en=1)
            dst_x = 0
            dst_y = 0
            # virtual_core_x = core_x // 2
            # virtual_core_y = core_y + core_x % 2 * 2
            virtual_core_x = core_x
            virtual_core_y = core_y
            A = virtual_core_x * 4 + virtual_core_y * 800
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, \
                                A_offset=28, Const=3, EN=1)
            if core_y in [0, 1]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 20 **************************************************#

    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 3 (50 * 256)  L3卷积 - 2 (48 * 256)
    if phase_en[22] == 1:
        assert(phase_en[14] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=8, py=6, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x6800 >> 2, addr_bias=0x0, \
                       addr_out=0x13200 >> 2)
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_18_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[17] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_18_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, in_cut_start=4, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xd540 >> 2, in_row_max=6)
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
                         addr_dout_base=0x380, addr_dout_length=799, send_num=1599, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1599,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if core_y % 2 == 0:
                if core_x == M - 1 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != M - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 1
                    router.Ny = 0
            else:
                if core_x == 0 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1599
                    router.Nx = -1
                    router.Ny = 0
            if core_y== 0 and core_x == 0:
                router.Receive_en = 0
                router.Send_en = 1
                router.CXY = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1599, A_offset=0, Const=0, EN=1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=5, py=10, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 21 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 4 (48 * 256)
    if phase_en[23] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if core_y in [2, 3]:
                length = 24 * 32
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0xbd20 >> 2,
                            addr_ciso=0x0 >> 2,
                            type_in=1, type_out=1, addr_out=0x9000)
                if phase_en[11] == 0:
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
            else:
                soma1 = None
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
                         addr_dout_base=0x1000, addr_dout_length=0, send_num=95, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1535, \
                         receive_num=15, t_mode=1, soma_in_en=1)
            dst_x = 0
            dst_y = 0
            # virtual_core_x = core_x // 2
            # virtual_core_y = core_y + core_x % 2 * 2
            virtual_core_x = core_x
            virtual_core_y = core_y
            A = virtual_core_x * 4 + virtual_core_y * 768
            if core_x - dst_x == 0 and core_y - dst_y == 0:
                router.Receive_en = 1
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=router.Send_number, \
                                A_offset=28, Const=3, EN=1)
            if core_y in [2, 3]:
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            else:
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Send_en = 0
                    map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 22 **************************************************#
    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 4 (48 * 256)  L3卷积 - 3 (50 * 256)
    if phase_en[24] == 1:
        #assert(phase_en[16] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=10, py=5, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x6800 >> 2, addr_bias=0x0, \
                       addr_out=0x13200 >> 2)
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_18_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[19] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_18_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=10, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, in_cut_start=4, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xdb40 >> 2, in_row_max=5)
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
                         addr_dout_base=0x380, addr_dout_length=767, send_num=1535, \
                         addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x380, addr_din_length=1535,\
                         receive_num=0, t_mode=1, soma_in_en=0)
            if core_y % 2 == 0:
                if core_x == M - 1 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != M - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 1
                    router.Ny = 0
            else:
                if core_x == 0 and core_y != N - 1:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = 0
                    router.Ny = 1
                elif core_x != 0:
                    router.CXY = 1
                    router.Relay_number = 1535
                    router.Nx = -1
                    router.Ny = 0
            if core_y== 0 and core_x == 0:
                router.Receive_en = 0
                router.Send_en = 1
                router.CXY = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                        cout=256, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                        in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000>>2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 23 **************************************************#
    # L3卷积 - 4 (48 * 256)
    if phase_en[25] == 1:
        #assert(phase_en[16] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=8, py=6, pad_on=False, load_bias=0, cin=256, cout=32, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2, addr_inb=0x6800 >> 2, addr_bias=0x0, \
                       addr_out=0x13200 >> 2)
            a = axon.init_data()
            axon.memory_blocks = [
                {'name': "core_{:d}_{:d}_phase_18_weight".format(core_x, core_y),
                 'start': axon.Addr_InB_base,
                 'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                 'data': a[1],
                 'mode': 0}
            ]
            if phase_en[21] == 0:
                axon.memory_blocks.append(
                    {'name': "core_{:d}_{:d}_phase_18_inX".format(core_x, core_y),
                     'start': axon.Addr_InA_base,
                     'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                     'data': a[0],
                     'mode': 0}
                )
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                        cout=32, px=8, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, in_cut_start=4, \
                        row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe180 >> 2, in_row_max=6)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 24 **************************************************#
    # L3层与shorcut结果求和，流水relu，保存在Mem0 0x1c400
    if phase_en[26] == 1:
        phase += 1
        for core in range(start_row * M, start_row * M + core_num):
            core_x = core % M
            core_y = core // M
            axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1, load_bias=0, \
                       cin=32, px=14, py=14, kx=2, ky=1, sx=1, sy=1, addr_in=0xcf00 >> 2, \
                       addr_bias=0x0, addr_out=0x10000 >> 2)
            if phase_en[22] == 0 or phase_en[9] == 0:
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
                        cout=32, px=14, py=14, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                        in_cut_start=1, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1c400 >> 2, in_row_max=14)
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
# phase_en = np.zeros(32).astype(int)
# # 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
# # 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
# # group间
# phase_en[0] = 0     # 组间
# phase_en[1] = 0     # 组间
# phase_en[2] = 0     # 组间
# phase_en[3] = 0     # 组间
#
# phase_en[4] = 0     # L1卷积计算，部分和收发------1 计算奇数行
# phase_en[5] = 0     # L1部分和求和，流水relu， ------1 奇数行
# phase_en[6] = 0     # L1卷积计算，部分和收发------2 计算偶数行
# phase_en[7] = 0     # L2部分和求和，流水relu， ------1 偶数行
# phase_en[8] = 0     # L1的结果通过路由整理成14 * 14 * 64 的数据
# phase_en[9] = 0     # L1多播，X1数据split，只保留32channel作为Xe
#
# phase_en[10] = 0    # L2卷积计算，部分和收发 ----- 1
# phase_en[11] = 0    # L2部分和求和，流水relu ----- 1
# phase_en[12] = 0    # L2卷积计算，部分和收发 ----- 2
# phase_en[13] = 0    # L2部分和求和，流水relu ----- 2
#
# phase_en[14] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 1 (50 * 256)
# phase_en[15] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 1 (50 * 256)
# phase_en[16] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 2 (48 * 256)
# phase_en[17] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 2 (48 * 256) L3卷积 - 1 (50 * 256) 保存在0x8800
# phase_en[18] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 3 (50 * 256)
# phase_en[19] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 3 (50 * 256) L3卷积 - 2 (48 * 256)
# phase_en[20] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 发送 - 4 (48 * 256)
# phase_en[21] = 1    # L2的结果整理 从14*14*32 -> 14*14*256 多播 - 4 (48 * 256) L3卷积 - 3 (50 * 256)
# phase_en[22] = 1    # L3卷积 - 4 (48 * 256)
#
# phase_en[23] = 1    # L3结果与Xe相加，流水relu，保存在Mem0 0x8800
#
#
#
#
# run = True  # 只生成map_config 还是生成后直接运行
#
# map_config = Gen_G11_Map_Config(phase_en, 28000, M=8 , N=4)
#
# from generator.test.Multi_Groups.AddRouterInfo import add_router_info
# map_config = add_router_info(map_config=map_config, group_idx_list=[0], chip_x_num=1, chip_y_num=1, core_x_num=16, core_y_num=10)
#
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