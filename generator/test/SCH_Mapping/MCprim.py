from primitive.Prim_41_Axon_CNN0_new import Prim_41_Axon
from primitive.Prim_81_Axon_CNN1_new import Prim_81_Axon
from primitive.Prim_06_move_merge_new import Prim_06_move_merge
from primitive.Prim_06_move_split_new import Prim_06_move_split
from primitive.Prim_09_Router import Prim_09_Router
from primitive.Prim_X5_Soma_compare_new import Prim_X5_Soma
from primitive.Prim_02_Axon_avgpooling_new import Prim_02_Axon
import numpy as np

def pX6(core_x, core_y, type, addr_in, addr_out, addr_ciso, length_in, num_in, length_ciso=16, num_ciso=None, \
        length_out=None, num_out=None, type_in=1, type_out=1, in_cut_start=0, reset_addr_in=1, reset_addr_ciso=1,\
        reset_addr_out=1, row_ck_on=0, in_row_max=None):
    soma1 = Prim_06_move_merge() if type == 6 else Prim_06_move_split()
    soma1.length_in = length_in
    soma1.length_ciso = length_ciso
    soma1.num_in = num_in
    soma1.num_ciso = num_in if num_ciso == None else num_ciso
    soma1.length_out = length_in if length_out == None else length_out
    soma1.num_out = num_in if num_out == None else num_out
    soma1.type_in = type_in
    soma1.type_out = type_out
    soma1.in_cut_start = in_cut_start
    soma1.Reset_Addr_in = reset_addr_in
    soma1.Reset_Addr_out = reset_addr_out
    soma1.Reset_Addr_ciso = reset_addr_ciso
    soma1.Row_ck_on = row_ck_on
    soma1.Addr_Start_in = addr_in
    soma1.Addr_Start_ciso = addr_ciso
    soma1.Addr_Start_out = addr_out
    soma1.in_row_max = in_row_max if in_row_max != None else (3 if soma1.Row_ck_on == 1 else 0)
    soma1.mem_sel = 1 if addr_out == 0x9000 else 0
    return soma1

def pX5(core_x, core_y, PIC_mode, addr_in, addr_out, cin, cout, px, py, kx, ky, sx, sy, cmp_c, pad_on=False,\
        pad_top=0, pad_down=0, pad_left=0, pad_right=0, type_in=1, type_out=1, in_cut_start=0, reset_addr_in=1,\
        reset_addr_out=1, row_ck_on=0, in_row_max=None):
    soma1 = Prim_X5_Soma()
    soma1.PIC_Mode = PIC_mode
    soma1.pad_on = pad_on
    soma1.type_in = type_in
    soma1.type_out = type_out
    soma1.cin = cin
    soma1.cout = cout
    soma1.Input_fm_Px = px
    soma1.Input_fm_Py = py
    soma1.pad_top = pad_top
    soma1.pad_down = pad_down
    soma1.pad_left = pad_left
    soma1.pad_right = pad_right
    soma1.pooling_Kx = kx
    soma1.pooling_Ky = ky
    soma1.pooling_Sx = sx
    soma1.pooling_Sy = sy
    soma1.CMP_C = cmp_c
    soma1.in_cut_start = in_cut_start
    soma1.reset_Addr_in = reset_addr_in
    soma1.reset_Addr_out = reset_addr_out
    soma1.Row_ck_on = row_ck_on
    soma1.Addr_Start_in = addr_in
    soma1.Addr_Start_out = addr_out
    soma1.in_row_max = in_row_max if in_row_max != None else (3 if soma1.Row_ck_on == 1 else 0)
    soma1.mem_sel = 1 if addr_out == 0x9000 else 0
    return soma1


def p09(core_x, core_y, rhead_mode, send_en, receive_en, send_num, addr_din_length, addr_rhead_base, \
        receive_num, addr_rhead_length, cxy=0, addr_dout_base=0x1000, addr_dout_length=None, addr_din_base=0x400,\
        nx=None, ny=None, relay_num=None, t_mode=1, soma_in_en=1):
    # 需要手动添加路由表
    router = Prim_09_Router()
    router.Rhead_mode = rhead_mode
    router.CXY = cxy
    router.Send_en = send_en
    router.Receive_en = receive_en
    router.Addr_Dout_base = addr_dout_base
    router.Dout_Mem_sel = 1 if addr_dout_base == 0x1000 else 0
    router.Addr_Dout_length = 0 if addr_dout_base == 0x1000 else addr_dout_length
    router.Send_number = send_num
    router.Addr_Rhead_base = addr_rhead_base
    router.Addr_Rhead_length = addr_rhead_length
    router.Addr_Din_base = addr_din_base
    router.Addr_Din_length = addr_din_length
    router.Receive_number = receive_num
    router.Nx = 0 if cxy == 0 else nx
    router.Ny = 0 if cxy == 0 else ny
    router.Send_PI_en = 0
    router.Back_sign_en = 0
    router.Send_PI_num = 0
    router.Receive_sign_num = 0
    router.Send_PI_addr_base = 0
    router.Relay_number = 0 if cxy == 0 else relay_num
    router.Q = 0
    router.Recevie_sign_en = 0
    router.T_mode = t_mode
    router.Soma_in_en = soma_in_en
    return router

def p41(core_x, core_y, px, py, cin, cout, kx, ky, sx, sy, addr_ina, addr_inb, addr_bias, addr_out, \
        ex=1, ey=1, ina_type=1, inb_type=1, load_bias=0, pad_on=False, pad_top=0, pad_down=0,\
        pad_right=0, pad_left=0, reset_addr_a=1, reset_addr_v=1, axon_delay=False, L5_num=None, L4_num=None, A2S2_mode=False ):
    axon = Prim_41_Axon()
    axon.pad_on = pad_on
    axon.InA_type = ina_type
    axon.InB_type = inb_type
    axon.Load_Bias = load_bias
    axon.cin = cin
    axon.cout = cout
    axon.Input_fm_Px = px
    axon.Input_fm_Py = py
    axon.pad_top = pad_top
    axon.pad_down = pad_down
    axon.pad_left = pad_left
    axon.pad_right = pad_right
    axon.conv_Kx = kx
    axon.conv_Ky = ky
    axon.conv_Sx = sx
    axon.conv_Sy = sy
    axon.conv_Ex = ex
    axon.conv_Ey = ey
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_ina
    axon.Addr_InB_base = addr_inb
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out
    axon.axon_delay = axon_delay
    if axon_delay:
        axon.L5_num = L5_num
        axon.L4_num = L4_num
    axon.A2S2_mode = A2S2_mode
    return axon

def p02(core_x, core_y, avf_pooling_en, pad_on, ina_type, load_bias, kx, ky, sx, sy, cin, px, py, addr_in, addr_bias,\
        addr_out, pad_top=0, pad_down=0, pad_left=0, pad_right=0, bias_length=0, reset_addr_a=1, reset_addr_v=1,):
    axon = Prim_02_Axon()
    axon.avg_pooling_en = avf_pooling_en
    axon.pad_on = pad_on
    axon.InA_type = ina_type
    axon.Load_Bias = load_bias
    axon.Bias_length = bias_length
    axon.cin = cin
    axon.Input_fm_Px = px
    axon.Input_fm_Py = py
    axon.pad_top = pad_top
    axon.pad_down = pad_down
    axon.pad_left = pad_left
    axon.pad_right = pad_right
    axon.pooling_Kx = kx
    axon.pooling_Ky = ky
    axon.pooling_Sx = sx
    axon.pooling_Sy = sy
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_in
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out
    axon.constant_b = 0
    return axon
