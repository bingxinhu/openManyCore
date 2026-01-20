import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G15_Map_Config(phase_en, clock, M, N, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
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
    for core in range(core_num):
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
    # 组间 发送5*512数据，接收10*256的数据
    if phase_en[0] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                if core_y == 0:
                    length = int(5 * 128)
                else:
                    length = 0
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=0x1c100 >> 2, addr_ciso=0x10000 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                ]
                if core_y in [0]:
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
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x3c0, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=1)
            if out_data_en == 1 and core_y in [0]:
                router.Send_en = 1
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_x_real = 8
                    dst_y = -1
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_x_real = 9
                    dst_y = -1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_x_real = 10
                    dst_y = -1
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_x_real = 11
                    dst_y = -1
                A = (core_x - dst_x) * 16 + core_y * 448
                if core_y == 0:
                    router.Send_number = int(5 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                # else:
                    # router.Send_number = int(4 * 128 / 8 - 1)
                    # router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                    #                 pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_y == 0 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 5
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 6
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 2 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 7
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 3 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6, 7]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 8
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x19000 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 组间 发送5*512数据，接收10*256的数据
    if phase_en[1] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                if core_y == 0:
                    length = int(2 * 128)
                    addr_in = 0x1c100 + 5*128 >> 2
                elif core_y == 1:
                    length = 3*128
                    addr_in = 0x1c100>> 2
                else:
                    length = 0
                    addr_in = 0x1c100>> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in, addr_ciso=0x10000 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                ]
                if core_y in [0, 1]:
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
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x3c0, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=1)
            if out_data_en == 1 and core_y in [0, 1]:
                router.Send_en = 1
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_x_real = 8
                    dst_y = -1
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_x_real = 9
                    dst_y = -1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_x_real = 10
                    dst_y = -1
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_x_real = 11
                    dst_y = -1
                A = (core_x - dst_x) * 16 + core_y * 128
                if core_y == 0:
                    router.Send_number = int(2 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                elif core_y == 1:
                    router.Send_number = int(3 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_y == 0 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 5
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 6
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 2 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 7
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 3 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6, 7]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 8
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x19000+10*256 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 组间 发送5*512数据，接收10*256的数据
    if phase_en[2] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                if core_y == 1:
                    length = int(4 * 128)
                    addr_in = 0x1c100 + 3*128 >> 2
                elif core_y == 2:
                    length = int(1 * 128)
                    addr_in = 0x1c100>> 2
                else:
                    length = 0
                    addr_in = 0x1c100>> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in, addr_ciso=0x10000 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                ]
                if core_y in [1, 2]:
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
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x3c0, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=1)
            if out_data_en == 1 and core_y in [1, 2]:
                router.Send_en = 1
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_x_real = 8
                    dst_y = -1
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_x_real = 9
                    dst_y = -1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_x_real = 10
                    dst_y = -1
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_x_real = 11
                    dst_y = -1
                A = (core_x - dst_x) * 16 + (core_y-1) * 256
                if core_y == 1:
                    router.Send_number = int(4 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                else:
                    router.Send_number = int(1 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_y == 0 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 5
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 6
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 2 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 7
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 3 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6, 7]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 8
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x19000+20*256 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 组间 发送10*512数据，接收19*256的数据
    if phase_en[3] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                if core_y == 2:
                    length = int(6 * 128)
                    addr_in = 0x1c100 + 1 * 128 >> 2
                elif core_y == 3:
                    length = int(4 * 128)
                    addr_in = 0x1c100 >> 2
                else:
                    length = 0
                    addr_in = 0x1c100 >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in, addr_ciso=0x10000 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
                ]
                if core_y in [2, 3]:
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
                    rhead_base = map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                        temp_phase - 1].Addr_Rhead_base
                    if map_config[0][0][((0, 0), (core_x, core_y))]['router'][temp_phase - 1].Send_en == 1:
                        rhead_base += (map_config[0][0][((0, 0), (core_x, core_y))]['router'][
                                           temp_phase - 1].Addr_Rhead_length + 1) * 4
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x3c0, addr_din_length=607, receive_num=7, t_mode=1, soma_in_en=1)
            if out_data_en == 1 and core_y in [2, 3]:
                router.Send_en = 1
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_x_real = 8
                    dst_y = -1
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_x_real = 9
                    dst_y = -1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_x_real = 10
                    dst_y = -1
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_x_real = 11
                    dst_y = -1
                A = (core_x - dst_x) * 16 + core_y * 384
                if core_y == 2:
                    router.Send_number = int(6 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
                elif core_y == 3:
                    router.Send_number = int(4 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=router.Send_number, A_offset=48, Const=15, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                if core_y == 0 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 5
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 6
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 2 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 7
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 3 and core_x != 15:
                    router.CXY = 1
                    router.Relay_number = router.Addr_Din_length
                    if core_x in [1, 2, 3, 4, 5, 6, 7]:
                        router.Nx = -1
                        router.Ny = 0
                    elif core_x == 0:
                        router.Nx = 8
                        router.Ny = 0
                    else:
                        router.Nx = 1
                        router.Ny = 0
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=256, \
                            cout=256, px=10, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x83c0, addr_out=0x19000+30*256 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 2 **************************************************#
    """
        组间 发送24*512数据
    """
    if phase_en[4] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if out_data_en == 1:
                if core_y == 0:
                    length = int(6 * 128)
                    addr_in = 0x1c480 >> 2
                elif core_y == 3:
                    length = int(8 * 128)
                    addr_in = 0x1c300 >> 2
                else:
                    length = int(5 * 128)
                    addr_in = 0x1c480 >> 2
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=length, length_ciso=16, length_out=length, \
                            num_in=1, num_ciso=1, num_out=1, row_ck_on=0, addr_in=addr_in, addr_ciso=0x0 >> 2,
                            addr_out=0x9000)
                s1 = soma1.init_data()
                soma1.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_1_soma1_in".format(core_x, core_y),
                     'start': soma1.Addr_Start_in,
                     'length': soma1.length_in * soma1.num_in // 4,
                     'data': s1[0],
                     'mode': 0},
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=0, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x3c0, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=1)
            # send
            if out_data_en == 1:
                router.Send_en = 1
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_x_real = 8
                    dst_y = -1
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_x_real = 9
                    dst_y = -1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_x_real = 10
                    dst_y = -1
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_x_real = 11
                    dst_y = -1
                if core_y == 0:
                    A = 192 + (core_x - dst_x) * 16 + core_y * 384
                    router.Send_number = int(6 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=95, A_offset=48, Const=15, EN=1)
                elif core_y == 3:
                    A = 192 + (core_x - dst_x) * 16 + 384 + (core_y - 1) * 320
                    router.Send_number = int(8 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=(core_x - dst_x) * 16,
                                    pack_per_Rhead=47, A_offset=48, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=79, A_offset=48, Const=15, EN=1)
                else:
                    A = 192 + (core_x - dst_x) * 16 + 384 + (core_y - 1) * 320
                    router.Send_number = int(5 * 128 / 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=79, A_offset=48, Const=15, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 3 **************************************************#
    """
        41卷积计算，部分和收发
    """
    if phase_en[5] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=5, pad_on=False, load_bias=0, cin=256, cout=32, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2, addr_inb=0x0, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_3_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                # if phase_en[0] == 0 or phase_en[1] == 0:
                #     axon.memory_blocks.append(
                #         {'name': "core_{:d}_{:d}_phase_3_inX".format(core_x, core_y),
                #          'start': axon.Addr_InA_base,
                #          'length': 256 * 7 * 7 // 4,    #axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                #          'data': a[0],
                #          'mode': 0}
                #     )

                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=5)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=559, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.Addr_Din_length = int(2 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 1:
                    router.Addr_Din_length = int(1 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 2:
                    router.Addr_Din_length = int(1 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 3:
                    router.Addr_Din_length = int(1 * 7 * 32 * 4 * 4 / 8 - 1)  # 8 需要减1
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=672, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=8 if core_y == 0 else 4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
    # ***************************************************** phase 4 **************************************************#
    """
        41部分和求和，为42层卷积计算传输数据
    """
    if phase_en[6] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0,\
                           cin=32, px=7, py=2 if core_y == 0 else 1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100>>2, \
                           addr_bias=0x0, addr_out=0xe000>>2)
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
                            cout=32, px=7, py=2 if core_y == 0 else 1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0,\
                            in_cut_start=2, row_ck_on=1, addr_in=0xe000 >> 2, addr_out=0x9000, in_row_max=2 if core_y\
                            == 0 else 1)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=55 if core_y == 0 else 27, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=559,\
                             receive_num=15, t_mode=1, soma_in_en=1)
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_y = 0
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_y = 1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_y = 2
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_y = 3
                if core_y == 0:
                    pack_per_rhead = 55
                    A = (core_x - dst_x) * 140
                else:
                    pack_per_rhead = 27
                    A = (core_x - dst_x) * 140 + 56 + (core_y - 1) * 28
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x-core_x, Y=dst_y-core_y, A=A, pack_per_Rhead=pack_per_rhead,\
                                A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    # ***************************************************** phase 25 **************************************************#
    """
        41部分和求和，为42层卷积计算传输数据
    """
    if phase_en[7] == 1:
        # assert(phase_en[3] == 1)# 需要与上一个phase一起跑
        phase += 1
        for core in range(core_num):
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
                             addr_dout_base=0x400, addr_dout_length=279, send_num=0, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=559,\
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 559
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=559, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 559
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 559
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=559, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 559
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 559
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=559, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 559
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 559
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=559, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 559
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

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=32, length_ciso=32, length_out=64, \
                            num_in=70, num_ciso=70, num_out=70, row_ck_on=0, addr_in=0x21000>>2, addr_ciso=0x218c0>>2,
                            type_in=1, type_out=1, addr_out=0x1c100>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 5 **************************************************#
    """
        41层输出数据按通道整理
    """
    if phase_en[8] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=64, length_ciso=64, length_out=128, \
                            num_in=35, num_ciso=35, num_out=35, row_ck_on=0, addr_in=0x1c100 >> 2,
                            addr_ciso=0x1c9c0 >> 2,
                            type_in=1, type_out=1, addr_out=0xe000 >> 2)
                if phase_en[24] == 0:
                    s1 = soma1.init_data()
                    soma1.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma1_in".format(core_x, core_y),
                         'start': soma1.Addr_Start_in,
                         'length': soma1.length_in * soma1.num_in // 4,
                         'data': s1[0],
                         'mode': 0},
                        {'name': "core_{:d}_{:d}_soma1_ciso".format(core_x, core_y),
                         'start': soma1.Addr_Start_ciso,
                         'length': soma1.length_ciso * soma1.num_ciso // 4,
                         'data': s1[1],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 6 **************************************************#
    """
        42层卷积计算
    """
    if phase_en[9] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=5, pad_on=True, pad_top=1, pad_down=0, pad_right=1,\
                           pad_left=1, load_bias=0, cin=128, cout=32, \
                           kx=3, ky=3, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0x10000>>2, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_6_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[4] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_6_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=4)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=447, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=447, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 3:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 7 **************************************************#
    """
        42部分和求和，为43层卷积计算传输数据
    """
    if phase_en[10] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                           load_bias=0, \
                           cin=32, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                           addr_bias=0x0, addr_out=0xe000 >> 2)
                if phase_en[5] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_7_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                            cout=32, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=2, row_ck_on=1, addr_in=0xe000 >> 2, addr_out=0x9000, in_row_max=1)
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
                #rhead_base = 0x30c #******************************************************************************-----//////
                router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=1, receive_en=0, cxy=0, \
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=27, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400,
                             addr_din_length=447, \
                             receive_num=15, t_mode=1, soma_in_en=1)
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_y = 0
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_y = 1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_y = 2
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_y = 3
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                pack_per_rhead = 27
                A = (core_x - dst_x) * 112 + (core_y) * 28
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                pack_per_Rhead=pack_per_rhead, \
                                A_offset=0, Const=0, EN=1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 26 **************************************************#
    """
        42部分和求和，为43层卷积计算传输数据
    """
    if phase_en[11] == 1:
        assert(phase_en[6] == 1)
        phase += 1
        for core in range(core_num):
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
                             addr_dout_base=0x400, addr_dout_length=223, send_num=0, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400,
                             addr_din_length=447, \
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 447
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

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=32, length_ciso=32, length_out=64, \
                            num_in=56, num_ciso=56, num_out=56, row_ck_on=0, addr_in=0x21000 >> 2,
                            addr_ciso=0x21700 >> 2,
                            type_in=1, type_out=1, addr_out=0xe000 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 8 **************************************************#
    """
        42层输出数据按通道整理，X1数据搬运到mem0
    """
    if phase_en[12] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=64, length_ciso=64, length_out=128, \
                            num_in=28, num_ciso=28, num_out=28, row_ck_on=0, addr_in=0xe000 >> 2,
                            addr_ciso=0xe700 >> 2,
                            type_in=1, type_out=1, addr_out=0x1f200 >> 2)
                if phase_en[6] == 0:
                    s1 = soma1.init_data()
                    soma1.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma1_in".format(core_x, core_y),
                         'start': soma1.Addr_Start_in,
                         'length': soma1.length_in * soma1.num_in // 4,
                         'data': s1[0],
                         'mode': 0},
                        {'name': "core_{:d}_{:d}_soma1_ciso".format(core_x, core_y),
                         'start': soma1.Addr_Start_ciso,
                         'length': soma1.length_ciso * soma1.num_ciso // 4,
                         'data': s1[1],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=1792, length_ciso=16, length_out=1792, \
                            num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x1ac00 >> 2,
                            addr_ciso=0xe000 >> 2,
                            type_in=1, type_out=1, addr_out=0xeb00 >> 2)
                if phase_en[1] == 0:
                    s2 = soma2.init_data()
                    soma2.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma2_in".format(core_x, core_y),
                         'start': soma2.Addr_Start_in,
                         'length': soma2.length_in * soma2.num_in // 4,
                         'data': s2[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 9 **************************************************#
    """
        43层卷积计算，部分和收发
    """
    if phase_en[13] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=4, pad_on=False, load_bias=0, cin=128, cout=128, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f200 >> 2, addr_inb=0xa000 >> 2, addr_bias=0x0, \
                           addr_out=0x1ac00 >> 2)
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
                        {'name': "core_{:d}_{:d}_phase_9_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1ac00 >> 2, addr_out=0x9000, in_row_max=4)
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
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=1343, receive_num=2, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1ba00 >> 2, in_row_max=0)
                elif core_y == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c800 >> 2, in_row_max=0)
                elif core_y == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1d600 >> 2, in_row_max=0)
                elif core_y == 3:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1ac00 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)


                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 10 **************************************************#
    """
        43层部分和求和，流水relu
    """
    if phase_en[14] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0 or core_y == 3:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1ac00 >> 2, \
                               addr_bias=0x0, addr_out=0x1e400 >> 2)
                elif core_y == 1:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1ba00 >> 2, \
                               addr_bias=0x0, addr_out=0x1ac00 >> 2)
                elif core_y == 2:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c800 >> 2, \
                               addr_bias=0x0, addr_out=0x1ac00 >> 2)
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
                            cout=128, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=2, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe000 >> 2, in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 11 **************************************************#
    """
        shortcut层卷积计算，部分和收发
    """
    if phase_en[15] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=4, pad_on=False, load_bias=0, cin=256, cout=128, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2, addr_inb=0x2000 >> 2, addr_bias=0x0, \
                           addr_out=0x1ac00 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_11_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[0] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_11_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1ac00 >> 2, addr_out=0x9000, in_row_max=4)
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
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=1343, receive_num=2, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1ba00 >> 2, in_row_max=0)
                elif core_y == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c800 >> 2, in_row_max=0)
                elif core_y == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=448, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1d600 >> 2, in_row_max=0)
                elif core_y == 3:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=896, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=0)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1ac00 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)


                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 12 **************************************************#
    """
        shortcut层部分和求和，流水relu
    """
    if phase_en[16] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0 or core_y == 3:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1ac00 >> 2, \
                               addr_bias=0x0, addr_out=0x1e400 >> 2)
                elif core_y == 1:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1ba00 >> 2, \
                               addr_bias=0x0, addr_out=0x1ac00 >> 2)
                elif core_y == 2:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c800 >> 2, \
                               addr_bias=0x0, addr_out=0x1ac00 >> 2)
                if phase_en[10] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_12_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                if core_y == 0:
                    addr_out = 0xe680 >> 2
                else:
                    addr_out = 0xe600 >> 2
                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=2, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=addr_out, in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 13 **************************************************#
    """
        #Y3与shorcut结果求和，保存在mem0#  
        同时将mem0中X1数据的后三行和mem1中的第4行合并，作为下一个循环的输入X1数据
    """
    if phase_en[17] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1, load_bias=0, \
                           cin=128, px=7, py=1, kx=2, ky=1, sx=1, sy=1, addr_in=0xe000 >> 2, \
                           addr_bias=0x0, addr_out=0x1ac00 >> 2)
                if phase_en[9] == 0 or phase_en[11] == 0:
                    assert(phase_en[9] == 0 and phase_en[11] == 0)  # 需要单独跑，或者三个phase一起跑
                    a = axon.init_data()
                    if phase_en[9] == 0:
                        axon.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_13_InA_1".format(core_x, core_y),
                             'start': 0xe000 >> 2,
                             'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky // 2,
                             'data': a[0][0:len(a[0])//2],
                             'mode': 0}
                        ]
                    if phase_en[11] == 0:
                        axon.memory_blocks = [
                            {'name': "core_{:d}_{:d}_phase_13_InA_2".format(core_x, core_y),
                             'start': 0xe380 >> 2,
                             'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky // 2,
                             'data': a[0][len(a[0])//2:len(a[0])],
                             'mode': 0}
                        ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe000 >> 2, in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=1792, length_ciso=16, length_out=1792, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0xeb00 >> 2, addr_ciso=0xe000 >> 2,
                            addr_out=0x1ac00 >> 2)
                if phase_en[7] == 0:
                    s2 = soma2.init_data()
                    soma2.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_13_soma2_in".format(core_x, core_y),
                         'start': soma2.Addr_Start_in,
                         'length': soma2.length_in * soma2.num_in // 4,
                         'data': s2[0],
                         'mode': 0},
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

# loop 2

    # ***************************************************** phase 14 **************************************************#
    """
        41卷积计算，部分和收发
    """
    if phase_en[18] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=4, pad_on=False, load_bias=0, cin=256, cout=32, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x1a500 >> 2, addr_inb=0x0, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_14_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[12] == 0 or phase_en[0] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_14_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )

                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=4)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=447, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=447, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 3:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
    # ***************************************************** phase 15 **************************************************#
    """
        41部分和求和，为42层卷积计算传输数据
    """
    if phase_en[19] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0,\
                           cin=32, px=7, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100>>2, \
                           addr_bias=0x0, addr_out=0xea80>>2)
                if phase_en[13] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_15_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                            cout=32, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0,\
                            in_cut_start=8, row_ck_on=1, addr_in=0xea80 >> 2, addr_out=0x9000, in_row_max=1)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=27, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=447,\
                             receive_num=15, t_mode=1, soma_in_en=1)
                if core_x in [0, 1, 2, 3]:
                    dst_x = 0
                    dst_y = 0
                elif core_x in [4, 5, 6, 7]:
                    dst_x = 4
                    dst_y = 1
                elif core_x in [8, 9, 10, 11]:
                    dst_x = 8
                    dst_y = 2
                elif core_x in [12, 13, 14, 15]:
                    dst_x = 12
                    dst_y = 3
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                pack_per_rhead = 27
                A = (core_x - dst_x) * 112 + (core_y) * 28
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x-core_x, Y=dst_y-core_y, A=A, pack_per_Rhead=pack_per_rhead,\
                                A_offset=0, Const=0, EN=1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 27 **************************************************#
    """
        41部分和求和，为42层卷积计算传输数据
    """
    if phase_en[20] == 1:
        assert(phase_en[14] == 1)
        phase += 1
        for core in range(core_num):
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
                             addr_dout_base=0x400, addr_dout_length=223, send_num=447, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400, addr_din_length=447,\
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 447
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 447
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 447
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

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=32, length_ciso=32, length_out=64, \
                            num_in=56, num_ciso=56, num_out=56, row_ck_on=0, addr_in=0x21000>>2, addr_ciso=0x21700>>2,
                            type_in=1, type_out=1, addr_out=0x1c100>>2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 16 **************************************************#
    """
        41层输出数据按通道整理
    """
    if phase_en[21] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=64, length_ciso=64, length_out=128, \
                            num_in=28, num_ciso=28, num_out=28, row_ck_on=0, addr_in=0x1c100 >> 2,
                            addr_ciso=0x1c800 >> 2,
                            type_in=1, type_out=1, addr_out=0xea80 >> 2)
                if phase_en[14] == 0:
                    s1 = soma1.init_data()
                    soma1.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma1_phase16_in".format(core_x, core_y),
                         'start': soma1.Addr_Start_in,
                         'length': soma1.length_in * soma1.num_in // 4,
                         'data': s1[0],
                         'mode': 0},
                        {'name': "core_{:d}_{:d}_soma1_phase16_ciso".format(core_x, core_y),
                         'start': soma1.Addr_Start_ciso,
                         'length': soma1.length_ciso * soma1.num_ciso // 4,
                         'data': s1[1],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 17 **************************************************#
    """
        42层卷积计算
    """
    if phase_en[22] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=4, pad_on=True, pad_top=0, pad_down=1, pad_right=1,\
                           pad_left=1, load_bias=0, cin=128, cout=32, \
                           kx=3, ky=3, sx=1, sy=1, addr_ina=0xea80 >> 2, addr_inb=0x10000>>2, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_17_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[15] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_17_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=3)
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=335, addr_rhead_base=rhead_base,\
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=447, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=112, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=224, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                elif core_y == 3:
                    router.Receive_en = 0
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=336, pack_per_Rhead=111, A_offset=0, Const=0, EN=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=32, \
                            cout=32, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                if core_y == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 18 **************************************************#
    """
        42部分和求和，为43层卷积计算传输数据
    """
    if phase_en[23] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                           load_bias=0, \
                           cin=32, px=7, py=1, kx=1, ky=4, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                           addr_bias=0x0, addr_out=0xea80 >> 2)
                if phase_en[16] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_18_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                if core_y == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                else:
                    map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=32, \
                            cout=32, px=7, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=8, row_ck_on=1, addr_in=0xea80 >> 2, addr_out=0x9000, in_row_max=1)
                if core_y == 3:
                    map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                else:
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
                             addr_dout_base=0x1000, addr_dout_length=0, send_num=27, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400,
                             addr_din_length=335, \
                             receive_num=11, t_mode=1, soma_in_en=1)
                if core_y < 3:
                    if core_x in [0, 1, 2, 3]:
                        dst_x = 0
                        dst_y = 0
                    elif core_x in [4, 5, 6, 7]:
                        dst_x = 4
                        dst_y = 1
                    elif core_x in [8, 9, 10, 11]:
                        dst_x = 8
                        dst_y = 2
                    elif core_x in [12, 13, 14, 15]:
                        dst_x = 12
                        dst_y = 3
                    pack_per_rhead = 27
                    A = (core_x - dst_x) * 84 + (core_y) * 28
                    router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=A,
                                    pack_per_Rhead=pack_per_rhead, \
                                    A_offset=0, Const=0, EN=1)
                if core_x - dst_x == 0 and core_y - dst_y == 0:
                    router.Receive_en = 1
                if core_y == 3:
                    router.Send_en = 0
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 28 **************************************************#
    """
        42部分和求和，为43层卷积计算传输数据
    """
    if phase_en[24] == 1:
        #assert(phase_en[17] == 1)
        phase += 1
        for core in range(core_num):
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
                             addr_dout_base=0x400, addr_dout_length=167, send_num=335, \
                             addr_rhead_base=rhead_base, addr_rhead_length=0, addr_din_base=0x400,
                             addr_din_length=335, \
                             receive_num=0, t_mode=1, soma_in_en=0)
                if core_y == 0:
                    if core_x == 0:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 335
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=335, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 335
                        router.Nx = 1
                        router.Ny = 0
                elif core_y == 1:
                    if core_x == 4:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 335
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=335, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 335
                        if core_x in [1, 2, 3]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 5
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 2:
                    if core_x == 8:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 335
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0,
                                        pack_per_Rhead=335, A_offset=0, Const=0, EN=1)
                    elif core_x != M - 1:
                        router.CXY = 1
                        router.Relay_number = 335
                        if core_x in [1, 2, 3, 4, 5, 6, 7]:
                            router.Nx = -1
                            router.Ny = 0
                        elif core_x == 0:
                            router.Nx = 9
                            router.Ny = 0
                        elif core_x != M - 1:
                            router.Nx = 1
                            router.Ny = 0
                elif core_y == 3:
                    if core_x == 12:
                        router.Receive_en = 0
                        router.Send_en = 1
                        router.Send_number = 335
                        router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                        pack_per_Rhead=335, A_offset=0, Const=0, EN=1)
                    elif core_x != 0:
                        router.CXY = 1
                        router.Relay_number = 335
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

                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=32, length_ciso=32, length_out=64, \
                            num_in=42, num_ciso=42, num_out=42, row_ck_on=0, addr_in=0x21000 >> 2,
                            addr_ciso=0x21540 >> 2,
                            type_in=1, type_out=1, addr_out=0xea80 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)


    # ***************************************************** phase 19 **************************************************#
    """
        42层输出数据按通道整理
    """
    if phase_en[25] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)

                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=64, length_ciso=64, length_out=128, \
                            num_in=21, num_ciso=21, num_out=21, row_ck_on=0, addr_in=0xea80 >> 2,
                            addr_ciso=0xe8c0 >> 2,
                            type_in=1, type_out=1, addr_out=0x19000 >> 2)
                if phase_en[17] == 0:
                    s1 = soma1.init_data()
                    soma1.memory_blocks = [
                        {'name': "core_{:d}_{:d}_soma1_in".format(core_x, core_y),
                         'start': soma1.Addr_Start_in,
                         'length': soma1.length_in * soma1.num_in // 4,
                         'data': s1[0],
                         'mode': 0},
                        {'name': "core_{:d}_{:d}_soma1_ciso".format(core_x, core_y),
                         'start': soma1.Addr_Start_ciso,
                         'length': soma1.length_ciso * soma1.num_ciso // 4,
                         'data': s1[1],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 20 **************************************************#
    """
        43层卷积计算，部分和收发
    """
    if phase_en[26] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=3, pad_on=False, load_bias=0, cin=128, cout=128, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2, addr_inb=0xa000 >> 2, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_20_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[18] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_20_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=3)
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
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.Addr_Din_length = int(6 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=6, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 1:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 2:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 3:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)


                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 21 **************************************************#
    """
        43层部分和求和，流水relu
    """
    if phase_en[27] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0, \
                               cin=128, px=6, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                else:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=5, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                if phase_en[19] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_21_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                if core_y == 0:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=6, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe380 >> 2, in_row_max=1)
                else:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                                cout=128, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                                in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe380 >> 2,
                                in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 22 **************************************************#
    """
        shortcut层卷积计算，部分和收发
    """
    if phase_en[28] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                axon = p41(core_x=core_x, core_y=core_y, px=7, py=3, pad_on=False, load_bias=0, cin=256, cout=128, \
                           kx=1, ky=1, sx=1, sy=1, addr_ina=0x1ac00 >> 2, addr_inb=0x2000 >> 2, addr_bias=0x0, \
                           addr_out=0x1c100 >> 2)
                a = axon.init_data()
                axon.memory_blocks = [
                    {'name': "core_{:d}_{:d}_phase_22_weight".format(core_x, core_y),
                     'start': axon.Addr_InB_base,
                     'length': axon.cin * axon.cout * axon.conv_Ky * axon.conv_Kx // 4,
                     'data': a[1],
                     'mode': 0}
                ]
                if phase_en[12] == 0:
                    axon.memory_blocks.append(
                        {'name': "core_{:d}_{:d}_phase_22_inX".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Py * axon.Input_fm_Px // 4,
                         'data': a[0],
                         'mode': 0}
                    )
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                            cout=128, px=7, py=3, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, in_cut_start=0, \
                            row_ck_on=1, addr_in=0x1c100 >> 2, addr_out=0x9000, in_row_max=3)
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
                             addr_rhead_length=1, addr_din_base=0x400, addr_din_length=0, receive_num=3, \
                             t_mode=1, soma_in_en=1)
                if core_y == 0:
                    router.Addr_Din_length = int(6 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=6, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 1:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=320, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 2:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=640, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                elif core_y == 3:
                    router.Addr_Din_length = int(5 * 128 * 4 * 4 // 8 - 1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=960, pack_per_Rhead=319, A_offset=0, Const=0, EN=1)
                    soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=0, cin=128, \
                                cout=128, px=5, py=4, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, \
                                in_cut_start=0, row_ck_on=0, addr_in=0x8400, addr_out=0x1c100 >> 2, in_row_max=0)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)


                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    # ***************************************************** phase 23 **************************************************#
    """
        shortcut层部分和求和，流水relu 
    """
    if phase_en[29] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0, load_bias=0, \
                               cin=128, px=6, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                else:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=0,
                               load_bias=0, \
                               cin=128, px=5, py=1, kx=2, ky=2, sx=1, sy=1, addr_in=0x1c100 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                if phase_en[21] == 0:
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_22_InA".format(core_x, core_y),
                         'start': axon.Addr_InA_base,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                if core_y == 0:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                            cout=128, px=6, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                            in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xea00 >> 2, in_row_max=1)
                else:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                                cout=128, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                                in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0xe980 >> 2,
                                in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    # ***************************************************** phase 24 **************************************************#
    """
        Y3与shorcut结果求和，保存在mem1 0x1c100
    """
    if phase_en[30] == 1:
        phase += 1
        for core in range(core_num):
            core_x = core % M
            core_y = core // M
            if core_y == N - 1 and N == 5:  # 最后一行core是不进行计算
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)
            else:  #
                if core_y == 0:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1, load_bias=0, \
                               cin=128, px=13, py=1, kx=2, ky=1, sx=1, sy=1, addr_in=0xe000 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                else:
                    axon = p02(core_x=core_x, core_y=core_y, avf_pooling_en=False, pad_on=False, ina_type=1,
                               load_bias=0, \
                               cin=128, px=12, py=1, kx=2, ky=1, sx=1, sy=1, addr_in=0xe000 >> 2, \
                               addr_bias=0x0, addr_out=0x19000 >> 2)
                if phase_en[20] == 0 or phase_en[22] == 0:
                    assert(phase_en[20] == 0 and phase_en[22] == 0)  # 需要单独跑，或者三个phase一起跑
                    a = axon.init_data()
                    axon.memory_blocks = [
                        {'name': "core_{:d}_{:d}_phase_24_InA".format(core_x, core_y),
                         'start': 0xe000 >> 2,
                         'length': axon.cin * axon.Input_fm_Px * axon.Input_fm_Py * axon.pooling_Ky * axon.pooling_Ky,
                         'data': a[0],
                         'mode': 0}
                    ]
                map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)

                if core_y == 0:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                                cout=128, px=13, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                                in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1c100 >> 2, in_row_max=1)
                else:
                    soma1 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=0, type_out=1, cin=128, \
                                cout=128, px=12, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, \
                                in_cut_start=8, row_ck_on=1, addr_in=axon.Addr_V_base, addr_out=0x1c100 >> 2,
                                in_row_max=1)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)

                map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)

                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    return map_config


# # ---------------------------------------------------------------------------------------#
# # 真正执行的代码
# phase_en = np.zeros(32)
# # 按下面的顺序依次代表每个phase是否运行（phase_en中的下标并不完全对应phase顺序）
# # 个别phase与其他phase有依赖关系，代码中用asser保证其相互依赖性
# # group间
# phase_en[0] = 0     # 接收4*7*256的数据，并横向多播
# phase_en[1] = 0     # 接收3*7*256的数据，并横向多播
# # loop 1
# phase_en[2] = 1     # 41卷积计算，部分和收发
# phase_en[3] = 0     # 41部分和求和，为42层卷积计算传输数据
# phase_en[24] = 0    # 5 #
# phase_en[4] = 0     # 41层输出数据按通道整理
#
# phase_en[5] = 0     # 42层卷积计算   --- 时钟较大 1.5k可以满足
# phase_en[6] = 0     # 42部分和求和，为43层卷积计算传输数据********
# phase_en[25] = 0
# phase_en[7] = 0     # 10 # 42层输出数据按通道整理，X1数据搬运到mem0
#
# phase_en[8] = 0     # 43层卷积，部分和收发
# phase_en[9] = 0     # 43层部分和求和，流水relu
# phase_en[10] = 0    # shortcut层卷积计算，部分和收发
# phase_en[11] = 0    # shortcut层部分和求和，流水relu
# phase_en[12] = 0    # 15 # Y3与shorcut结果求和，保存在mem0，同时将mem0中X1数据的后三行和mem1中的第4行合并，作为下一个循环的输入X1数据
# # loop 2
# phase_en[13] = 0     # 41卷积计算，部分和收发
# phase_en[14] = 0     # 41部分和求和，为42层卷积计算传输数据
# phase_en[26] = 0
# phase_en[15] = 0     # 41层输出数据按通道整理
#
# phase_en[16] = 0     # 20 # 42层卷积计算   --- 时钟较大 1.5k可以满足
# phase_en[17] = 0     # 42部分和求和，为43层卷积计算传输数据
# phase_en[27] = 0
# phase_en[18] = 0     # 42层输出数据按通道整理
#
# phase_en[19] = 0     # 43层卷积，部分和收发       ************************************一个地址多次覆盖写的话，python只会打印一次，c++会打印多次
# phase_en[20] = 0     # 25 # 43层部分和求和，流水relu
# phase_en[21] = 0    # shortcut层卷积计算，部分和收发   ************************************一个地址多次覆盖写的话，python只会打印一次，c++会打印多次
# phase_en[22] = 0    # shortcut层部分和求和，流水relu
# phase_en[23] = 0    # Y3与shorcut结果求和，保存在mem0
#
# run = True  # 只生成map_config 还是生成后直接运行
#
# map_config = Gen_G15_Map_Config(phase_en, 15000, M=1, N=4)
#
# from generator.test.Multi_Groups.AddRouterInfo import add_router_info
# map_config = add_router_info(map_config=map_config, group_idx_list=[0], chip_x_num=1, chip_y_num=1, core_x_num=16, core_y_num=10)
#
#
# # import pickle
# # with open('G15_64cores\\G15_64cores_phase_1_28.map_config', 'wb') as f:
# #     pickle.dump(map_config, f)
#
# if run:
#     from generator.test_engine import TestMode, TestEngine
#
#     test_phase = []
#     for i in range(len(map_config[0][0][((0, 0), (0, 0))]['axon'])):
#         test_phase.append((0, i + 1))
#     map_config['sim_clock'] = min(len(map_config[0][0][((0, 0), (0, 0))]['axon']) * map_config[0][0]['clock'] - 1, 100000)
#     test_config = {
#             'tb_name': 'M99999',
#             'test_mode': TestMode.MEMORY_STATE,
#             'test_group_phase': test_phase
#         }
#
#     tester = TestEngine(map_config, test_config)
#     assert tester.run_test()