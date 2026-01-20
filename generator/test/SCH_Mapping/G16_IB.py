import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G16_IB(phase_en, clock, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group16 Input Buffer
    4 cores
    chip(2, 0)
    (0, 9) (4, 9) (8, 9) (12, 9)
    前两个phase发送，后两个phase接收
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
    for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # 发送5 * 512的数据， 接收3*512
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0 >> 2, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0 >> 2, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 11994
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x19000 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=319, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=191, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Soma_in_en = 1
                router.Send_en =  1
                if (core_x, core_y)  == (0, 9):
                    dst_x = 0
                    dst_y = 5
                elif (core_x, core_y)  == (4, 9):
                    dst_x = 4
                    dst_y = 6
                elif (core_x, core_y)  == (8, 9):
                    dst_x = 8
                    dst_y = 7
                elif (core_x, core_y)  == (12, 9):
                    dst_x = 12
                    dst_y = 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=3, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 发送5 * 512的数据， 接收3*512
    if phase_en[1] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0 >> 2, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0 >> 2, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 11994
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x19000+5*512 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=319, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=191, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Soma_in_en = 1
                router.Send_en =  1
                if (core_x, core_y)  == (0, 9):
                    dst_x = 0
                    dst_y = 5
                elif (core_x, core_y)  == (4, 9):
                    dst_x = 4
                    dst_y = 6
                elif (core_x, core_y)  == (8, 9):
                    dst_x = 8
                    dst_y = 7
                elif (core_x, core_y)  == (12, 9):
                    dst_x = 12
                    dst_y = 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=3, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+3*512 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 发送5 * 512的数据， 接收3*512
    if phase_en[2] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0 >> 2, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0 >> 2, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 11994
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x19000+10*512 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=319, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=191, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Soma_in_en = 1
                router.Send_en =  1
                if (core_x, core_y)  == (0, 9):
                    dst_x = 0
                    dst_y = 5
                elif (core_x, core_y)  == (4, 9):
                    dst_x = 4
                    dst_y = 6
                elif (core_x, core_y)  == (8, 9):
                    dst_x = 8
                    dst_y = 7
                elif (core_x, core_y)  == (12, 9):
                    dst_x = 12
                    dst_y = 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=3, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+6*512 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 1 **************************************************#
    # 发送10 * 512的数据， 接收9*512
    if phase_en[3] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0 >> 2, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0 >> 2, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 11994
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=512, length_ciso=16,
                            length_out=512, \
                            num_in=10, num_ciso=10, num_out=10, row_ck_on=0, addr_in=0x19000+15*512 >> 2,
                            addr_ciso=0x10000 >> 2,
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
                         addr_din_base=0x380, addr_din_length=575, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Soma_in_en = 1
                router.Send_en =  1
                if (core_x, core_y)  == (0, 9):
                    dst_x = 0
                    dst_y = 5
                elif (core_x, core_y)  == (4, 9):
                    dst_x = 4
                    dst_y = 6
                elif (core_x, core_y)  == (8, 9):
                    dst_x = 8
                    dst_y = 7
                elif (core_x, core_y)  == (12, 9):
                    dst_x = 12
                    dst_y = 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=3, py=1, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=0x19000+9*512 >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 2 **************************************************#
    """
    发送24 * 512的数据，接收10*512
    """
    if phase_en[4] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0 >> 2, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0 >> 2, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 11994
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=3072, length_ciso=16,
                            length_out=3072, \
                            num_in=4, num_ciso=4, num_out=4, row_ck_on=0, addr_in=0x1c200 >> 2,
                            addr_ciso=0X10000 >> 2,
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
                         addr_dout_length=0, send_num=1535, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=1)
            if out_data_en == 1:
                router.Send_en = 1
                if (core_x, core_y)  == (0, 9):
                    dst_x = 0
                    dst_y = 5
                elif (core_x, core_y)  == (4, 9):
                    dst_x = 4
                    dst_y = 6
                elif (core_x, core_y)  == (8, 9):
                    dst_x = 8
                    dst_y = 7
                elif (core_x, core_y)  == (12, 9):
                    dst_x = 12
                    dst_y = 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = 639
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=2, py=5, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x19000 + 18*512) >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)

    #***************************************************** phase 3 **************************************************#
    """
    接收21 * 512的数据
    """
    if phase_en[5] == 1:
        phase += 1
        for (core_x, core_y) in [(0, 9), (4, 9), (8, 9), (12, 9)]:
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
                         addr_dout_length=0, send_num=1599, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1599, receive_num=0, t_mode=1, soma_in_en=0)
            # receive
            if in_data_en == 1:
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = 1343
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX5(core_x=core_x, core_y=core_y, PIC_mode=0, pad_on=False, type_in=1, type_out=1, cin=512, \
                            cout=512, px=3, py=7, kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, \
                            in_cut_start=0, row_ck_on=0, addr_in=0x8380, addr_out=(0x19000 + 28 * 512) >> 2)
                map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)
            else:
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