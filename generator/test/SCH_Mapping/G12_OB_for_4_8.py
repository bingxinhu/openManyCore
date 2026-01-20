import numpy as np
from generator.test.SCH_Mapping.MCprim import pX6, p09, pX5, p41, p02

def Gen_G12_OB(phase_en, clock,start_row=0, in_data_en=0, out_data_en=0, delay_L4=(), delay_L5=()):
    """
    ResNet-50 Group12 Output Buffer Mapping
    chip (1, 2)
    (12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)
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
    for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
        map_config[0][0][((0, 0), (core_x, core_y))] = {
            'axon': [],
            'soma1': [],
            'router': [],
            'soma2': []
        }
    phase = -1 # 当前Phase数 从0开始
    #***************************************************** phase 1 **************************************************#
    # 接收 5 * 256  ，发送7*256
    if phase_en[0] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[0], L5_num=delay_L5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=7, num_ciso=7, num_out=7, row_ck_on=0, addr_in=0x10000 >> 2,
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
                         addr_dout_length=0, send_num=223, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=159, receive_num=7, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=0x16200 >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 1 **************************************************#
    # 接收 5 * 256  ，发送7*256
    if phase_en[1] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[1], L5_num=delay_L5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=7, num_ciso=7, num_out=7, row_ck_on=0, addr_in=0x10000+7*256 >> 2,
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
                         addr_dout_length=0, send_num=223, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=159, receive_num=7, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=0x16200+5*256 >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 1 **************************************************#
    # 接收 5 * 256  ，发送7*256
    if phase_en[2] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[2], L5_num=delay_L5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=7, num_ciso=7, num_out=7, row_ck_on=0, addr_in=0x10000+14*256 >> 2,
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
                         addr_dout_length=0, send_num=223, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=159, receive_num=7, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=5, num_ciso=5, num_out=5, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=0x16200+10*256 >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 1 **************************************************#
    # 接收 10 * 256  ，发送15*256
    if phase_en[3] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[3], L5_num=delay_L5[3], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=15, num_ciso=15, num_out=15, row_ck_on=0, addr_in=0x10000+21*256 >> 2,
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
                         addr_dout_length=0, send_num=479, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=319, receive_num=7, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                    pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=10, num_ciso=10, num_out=10, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=0x16200+15*256 >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 2 **************************************************#
    # 接收 25 * 256  发送20*256
    if phase_en[4] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[4], L5_num=delay_L5[4], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=20, num_ciso=20, num_out=20, row_ck_on=0, addr_in=(0x10000 + 36*256) >> 2,
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
                         addr_dout_length=0, send_num=639, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=799, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                router.Receive_number = 7
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=25, num_ciso=25, num_out=25, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x16200 + 25*256) >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 3 **************************************************#
    # 接收 24 * 256  发送20*256
    if phase_en[5] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(core_x=core_x, core_y=core_y, px=1, py=1, pad_on=False, load_bias=0, cin=1, cout=1, \
                       kx=1, ky=1, sx=1, sy=1, addr_ina=0x0, addr_inb=0x0, addr_bias=0x0, \
                       addr_out=0x0, axon_delay=True, L4_num=delay_L4[5], L5_num=delay_L5[5], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            # 13363
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(axon)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=20, num_ciso=20, num_out=20, row_ck_on=0, addr_in=(0x10000 + 56*256) >> 2,
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
                         addr_dout_length=0, send_num=639, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=767, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                router.Receive_number = 7
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=24, num_ciso=24, num_out=24, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x16200 + 50*256) >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 4 **************************************************#
    # 接收 24 * 256 ，发送22*256
    if phase_en[6] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            if out_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=22, num_ciso=22, num_out=22, row_ck_on=0, addr_in=(0x10000 + 76*256) >> 2,
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
                         addr_dout_length=0, send_num=703, addr_rhead_base=rhead_base, addr_rhead_length=0, cxy=0, \
                         addr_din_base=0x380, addr_din_length=767, receive_num=0, t_mode=1, soma_in_en=0)
            if out_data_en == 1:
                router.Send_en = 1
                router.Soma_in_en = 1
                if core_y == 0:
                    dst_x = core_x
                    dst_y = -1
                else:
                    dst_x = 16 + 5 + (core_x - 12) * 2
                    dst_y = 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                pack_per_Rhead=router.Send_number, A_offset=0, Const=0, EN=1)
            if in_data_en == 1:
                router.Receive_en = 1
                router.Receive_number = 7
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(router)

            if in_data_en == 1:
                soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                            length_out=256, \
                            num_in=24, num_ciso=24, num_out=24, row_ck_on=0, addr_in=0x8380,
                            addr_ciso=0x10000 >> 2,
                            addr_out=(0x16200 + 74 * 256) >> 2)
            else:
                soma1 = None
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(soma1)

    #***************************************************** phase 5 **************************************************#
    # 搬运 0x16200 -> 0x10000
    if phase_en[7] == 1:
        phase += 1
        for (core_x, core_y) in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            soma1 = pX6(core_x=core_x, core_y=core_y, type=6, length_in=256, length_ciso=16,
                        length_out=256, \
                        num_in=98, num_ciso=98, num_out=98, row_ck_on=0, addr_in=0x16200 >> 2,
                        addr_ciso=0x10000 >> 2,
                        addr_out=0x10000 >> 2)
            map_config[0][0][((0, 0), (core_x, core_y))]['axon'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma1'].append(soma1)
            map_config[0][0][((0, 0), (core_x, core_y))]['router'].append(None)
            map_config[0][0][((0, 0), (core_x, core_y))]['soma2'].append(None)


    return map_config