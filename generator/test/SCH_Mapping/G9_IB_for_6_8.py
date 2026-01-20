import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G9_IB(phase_en, clock,start_row=0, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group9 Input Buffer Mapping
    chip (1, 2)
    (8, 0), (9, 0), (10, 0), (11, 0)
    前4个phase发送
    后4个phase接收
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
    for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # 发送 20 * 128  接收18*128
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=20, num_ciso=20, num_out=20, row_ck_on=0, addr_in=0x10000 >> 2,
                            addr_ciso=0x1c100 >> 2,
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

            router = p09(core_x=core_x, core_y=core_y, rhead_mode=1, send_en=0, receive_en=0, addr_dout_base=0x1000, \
                         addr_dout_length=0, send_num=319, addr_rhead_base=0x300, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_x == 8:
                    nx = -5
                    ny = 0
                elif core_x == 9:
                    nx = -5
                    ny = 0
                elif core_x == 10:
                    nx = -5
                    ny = 0
                elif core_x == 11:
                    nx = -5
                    ny = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                piexl_num = 18
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=0x18000 >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 2 **************************************************#
    # 发送 20 * 128  接收18*128
    if phase_en[1] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=20, num_ciso=20, num_out=20, row_ck_on=0, addr_in=(0x10000+20*128) >> 2,
                            addr_ciso=0x1c100 >> 2,
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
                         addr_dout_length=0, send_num=319, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_x == 8:
                    nx = -5
                    ny = 0
                elif core_x == 9:
                    nx = -5
                    ny = 0
                elif core_x == 10:
                    nx = -5
                    ny = 0
                elif core_x == 11:
                    nx = -5
                    ny = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                piexl_num = 18
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x18000 + 18*128) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 3 **************************************************#
    # 发送 20 * 128  接收18*128
    if phase_en[2] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=20, num_ciso=20, num_out=20, row_ck_on=0, addr_in=(0x10000+40*128) >> 2,
                            addr_ciso=0x1c100 >> 2,
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
                         addr_dout_length=0, send_num=319, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_x == 8:
                    nx = -5
                    ny = 0
                elif core_x == 9:
                    nx = -5
                    ny = 0
                elif core_x == 10:
                    nx = -5
                    ny = 0
                elif core_x == 11:
                    nx = -5
                    ny = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                piexl_num = 18
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x18000 + 36 * 128) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 4 **************************************************#
    # 发送 38 * 128  接收47*128
    if phase_en[3] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=38, num_ciso=38, num_out=38, row_ck_on=0, addr_in=(0x10000+60*128) >> 2,
                            addr_ciso=0x1c100 >> 2,
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
                         addr_dout_length=0, send_num=607, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=0, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_x == 8:
                    nx = -5
                    ny = 0
                elif core_x == 9:
                    nx = -5
                    ny = 0
                elif core_x == 10:
                    nx = -5
                    ny = 0
                elif core_x == 11:
                    nx = -5
                    ny = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=nx, Y=ny, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                piexl_num = 47
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x18000 + 18 * 3 * 128) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 5 **************************************************#
    # 发送 98 * 128  接收47*128
    if phase_en[4] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=98, num_ciso=98, num_out=98, row_ck_on=0, addr_in=0x13100 >> 2,
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
                         addr_dout_length=0, send_num=1567, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1567, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_x == 8:
                    nx = -5
                    ny = 0
                elif core_x == 9:
                    nx = -5
                    ny = 0
                elif core_x == 10:
                    nx = -5
                    ny = 0
                elif core_x == 11:
                    nx = -5
                    ny = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=nx, Y=ny, A=0,
                                pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                piexl_num = 47
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x18000 + (18 * 3 + 47) * 128) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 6 **************************************************#
    #  接收48*128
    if phase_en[5] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
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
                         addr_dout_length=0, send_num=1567, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=1567, receive_num=0, t_mode=1, soma_in_en=0)
            if in_data_en == 1:
                piexl_num = 48
                router.Receive_en = 1
                router.Receive_number = 0
                router.Addr_Din_length = piexl_num * 128 // 8 - 1
                router.recv_source_core_grp = [{'core_id': [((1, 1), (core_x, 9))],
                                                'data_num': router.Addr_Din_length + 1,
                                                'T_mode': 1,
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase
                                                }
                                               ]
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                            length_out=128, \
                            num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x18000 + (18 * 3 + 47 * 2) * 128) >> 2)
            else:
                soma2 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    #***************************************************** phase 7 **************************************************#
    #  搬运
    if phase_en[6] == 1:
        phase += 1
        for (core_x, core_y) in [(8, 0), (9, 0), (10, 0), (11, 0)]:
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            piexl_num = 196
            soma2 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=128, length_ciso=16,
                        length_out=128, \
                        num_in=piexl_num, num_ciso=piexl_num, num_out=piexl_num, row_ck_on=0, addr_in=0x18000 >> 2,
                        addr_ciso=0x10000 >> 2,
                        addr_out=0x10000 >> 2)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma2)

    return map_config


