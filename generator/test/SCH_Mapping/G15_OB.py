import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G15_OB(phase_en, clock, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group15 Output Buffer
    4 cores
    chip(2, 1)
    (0, 0), (4, 0), (8, 0), (12, 0)
    前两个phase接收数据，后两个phase发送数据
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
    for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # 接收5 * 512的数据  发送 3 * 512
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x19000 >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=3, t_mode=1, soma_in_en=1)
            if in_data_en == 1:
                router.Receive_en = 1
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 接收5 * 512的数据  发送 3 * 512
    if phase_en[1] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x19000+3*512 >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=1)
            if in_data_en == 1:
                router.Receive_en = 1
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+5*512 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 接收5 * 512的数据  发送 3 * 512
    if phase_en[2] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=3, num_ciso=3, num_out=3, row_ck_on=0, addr_in=0x19000+6*512 >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=191, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=1)
            if in_data_en == 1:
                router.Receive_en = 1
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+10*512 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 1 **************************************************#
    # 接收10 * 512的数据  发送 9 * 512
    if phase_en[3] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=9, num_ciso=9, num_out=9, row_ck_on=0, addr_in=0x19000+9*512 >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=575, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=639, receive_num=7, t_mode=1, soma_in_en=1)
            if in_data_en == 1:
                router.Receive_en = 1
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=5, py=2, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x10000+15*512 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 2 **************************************************#
    """
    接收24 * 512的数据  发送10*512 数据
    """
    if phase_en[4] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0xe000 >> 2, addr_inb=0xe000 >> 2, addr_bias=0xe000 >> 2, \
                       addr_out=0xe000 >> 2, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 17914
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=1024, length_ciso=16,
                            length_out=1024, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=(0x19000 + 18*512) >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=639, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1535, receive_num=19, t_mode=1, soma_in_en=1)
            if in_data_en == 1:
                router.Receive_en = 1
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)
            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=4, py=6, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080,\
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x10000 + 25*512) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 3 **************************************************#
    """
    发送21 * 512的数据
    """
    if phase_en[5] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=21, num_ciso=21, num_out=21, row_ck_on=0, addr_in=(0x19000 + 28 * 512) >> 2,
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
            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=1343, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=15, t_mode=1, soma_in_en=1)
            if out_data_en == 1:
                router.Send_en = 1
                nx = 0
                ny = -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=nx, Y=ny, A=0,
                                pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

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