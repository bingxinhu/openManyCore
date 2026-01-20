import torch
import torch.nn as nn
from math import sqrt
import copy

def cut(x, in_cut_start: int = 4, en=True, type_out: int = 1):
    """
    result = (x >> (2 * in_cut_start).clip()
    type_out: 0-int32, 1-int8
    """
    if en:
        in_cut_start = in_cut_start
        type_out = type_out
        if type_out == 0:
            qmax, qmin = 0x7fffffff, -0x80000000
        elif type_out == 1:
            qmax, qmin = 0x7f, -0x80
        else:
            raise ValueError
        return x.div(2 ** (2 * in_cut_start)).floor().clamp(min=qmin, max=qmax)
    else:
        return x


class GRUCell(nn.Module):
    """自定义GRUCell"""
    def __init__(self, input_size, hidden_size, quantization_en):
        super(GRUCell, self).__init__()
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置) 
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
        lb, ub = -sqrt(1/hidden_size), sqrt(1/hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.grucell_summary = {}
        self.in_cut_start_list = {}
        self.lut_list = {}
        self.q_one = 127
        self.quantization_en = quantization_en

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    def forward(self, x, hid):
        GRU1 = torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0]
        self.grucell_summary['GRU1'] = {'input': x.squeeze(0).detach(),
                                        'weight': self.in2hid_w[0].data.t(),
                                        'bias': self.in2hid_b[0].data,
                                        'output': GRU1.squeeze(0).detach()}
        GRU2 = torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0]
        self.grucell_summary['GRU2'] = {'input': hid.squeeze(0).detach(),
                                        'weight': self.hid2hid_w[0].data.t(),
                                        'bias': self.hid2hid_b[0].data,
                                        'output': GRU2.squeeze(0).detach()}
        GRU3 = GRU1 + GRU2
        self.grucell_summary['GRU3'] = {'input1': GRU1.squeeze(0).detach(),
                                        'input2': GRU2.squeeze(0).detach(),
                                        'output': GRU3.squeeze(0).detach()}
        GRU3_cut = cut(GRU3, in_cut_start=self.in_cut_start_list['GRU3_cut'], en=self.quantization_en)
        self.grucell_summary['GRU3_cut'] = {'input': GRU3.squeeze(0).detach(),
                                            'output': GRU3_cut.squeeze(0).detach()}
        r = torch.floor(self.lut_list['sigmoid_r_m'] * torch.sigmoid(GRU3_cut / self.lut_list['sigmoid_r_d']))
        self.grucell_summary['r'] = {'input': GRU3_cut.squeeze(0).detach(),
                                     'output': r.squeeze(0).detach()}
        GRU4 = torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1]
        self.grucell_summary['GRU4'] = {'input': x.squeeze(0).detach(),
                                        'weight': self.in2hid_w[1].data.t(),
                                        'bias': self.in2hid_b[1].data,
                                        'output': GRU4.squeeze(0).detach()}
        GRU5 = torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1]
        self.grucell_summary['GRU5'] = {'input': hid.squeeze(0).detach(),
                                        'weight': self.hid2hid_w[1].data.t(),
                                        'bias': self.hid2hid_b[1].data,
                                        'output': GRU5.squeeze(0).detach()}
        GRU6 = GRU4 + GRU5
        self.grucell_summary['GRU6'] = {'input1': GRU4.squeeze(0).detach(),
                                        'input2': GRU5.squeeze(0).detach(),
                                        'output': GRU6.squeeze(0).detach()}
        GRU6_cut = cut(GRU6, in_cut_start=self.in_cut_start_list['GRU6_cut'], en=self.quantization_en)
        self.grucell_summary['GRU6_cut'] = {'input': GRU6.squeeze(0).detach(),
                                            'output': GRU6_cut.squeeze(0).detach()}
        z = torch.floor(self.lut_list['sigmoid_z_m'] * torch.sigmoid(GRU6_cut / self.lut_list['sigmoid_z_d']))
        self.grucell_summary['z'] = {'input': GRU6_cut.squeeze(0).detach(),
                                     'output': z.squeeze(0).detach()}
        GRU7 = torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2]
        self.grucell_summary['GRU7'] = {'input': x.squeeze(0).detach(),
                                        'weight': self.in2hid_w[2].data.t(),
                                        'bias': self.in2hid_b[2].data,
                                        'output': GRU7.squeeze(0).detach()}
        GRU8 = torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2]
        self.grucell_summary['GRU8'] = {'input': hid.squeeze(0).detach(),
                                        'weight': self.hid2hid_w[2].data.t(),
                                        'bias': self.hid2hid_b[2].data,
                                        'output': GRU8.squeeze(0).detach()}
        GRU8_cut = cut(GRU8, in_cut_start=self.in_cut_start_list['GRU8_cut'], en=self.quantization_en)
        self.grucell_summary['GRU8_cut'] = {'input': GRU8.squeeze(0).detach(),
                                            'output': GRU8_cut.squeeze(0).detach()}
        GRU9 = torch.mul(r, GRU8_cut)
        self.grucell_summary['GRU9'] = {'input1': r.squeeze(0).detach(),
                                        'input2': GRU8_cut.squeeze(0).detach(),
                                        'output': GRU9.squeeze(0).detach()}
        GRU10 = GRU7 + GRU9
        self.grucell_summary['GRU10'] = {'input1': GRU7.squeeze(0).detach(),
                                         'input2': GRU9.squeeze(0).detach(),
                                         'output': GRU10.squeeze(0).detach()}
        GRU10_cut = cut(GRU10, in_cut_start=self.in_cut_start_list['GRU10_cut'], en=self.quantization_en)
        self.grucell_summary['GRU10_cut'] = {'input': GRU10.squeeze(0).detach(),
                                             'output': GRU10_cut.squeeze(0).detach()}
        n = torch.floor(self.lut_list['tanh_m'] * torch.tanh(GRU10_cut / self.lut_list['tanh_d']))
        self.grucell_summary['n'] = {'input': GRU10_cut.squeeze(0).detach(),
                                     'output': n.squeeze(0).detach()}
        GRU11 = self.q_one - z
        self.grucell_summary['GRU11'] = {'input': z.squeeze(0).detach(),
                                         'output': GRU11.squeeze(0).detach()}
        GRU11_cut = cut(GRU11, in_cut_start=self.in_cut_start_list['GRU11_cut'], en=self.quantization_en)
        self.grucell_summary['GRU11_cut'] = {'input': GRU11.squeeze(0).detach(),
                                             'output': GRU11_cut.squeeze(0).detach()}
        GRU12 = torch.mul(n, GRU11_cut)
        self.grucell_summary['GRU12'] = {'input1': n.squeeze(0).detach(),
                                         'input2': GRU11_cut.squeeze(0).detach(),
                                         'output': GRU12.squeeze(0).detach()}
        GRU13 = torch.mul(hid, z)
        self.grucell_summary['GRU13'] = {'input1': hid.squeeze(0).detach(),
                                         'input2': z.squeeze(0).detach(),
                                         'output': GRU13.squeeze(0).detach()}
        next_hid = GRU12 + GRU13
        self.grucell_summary['next_hid'] = {'input1': GRU12.squeeze(0).detach(),
                                            'input2': GRU13.squeeze(0).detach(),
                                            'output': next_hid.squeeze(0).detach()}      
        next_hid_cut = cut(next_hid, in_cut_start=self.in_cut_start_list['next_hid_cut'], en=self.quantization_en) 
        self.grucell_summary['next_hid_cut'] = {'input': next_hid.squeeze(0).detach(),
                                                'output': next_hid_cut.squeeze(0).detach()}
        # r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
        #                   torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        # z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
        #                   torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        # n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
        #                torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
        # next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
        return next_hid_cut


class GRU(nn.Module):   
    def __init__(self, input_num, hidden_num, in_cut_start_mat, q_one_list, lut_mat, quantization_en=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = GRUCell(input_num, hidden_num, quantization_en=quantization_en)
        self.grucell_back = GRUCell(input_num, hidden_num, quantization_en=quantization_en)
        self.gru_summary = {}
        self.in_cut_start_mat = in_cut_start_mat
        self.q_one_list = q_one_list
        self.lut_mat = lut_mat

    def forward(self, x, hid=None):
        assert x.shape[1] == len(self.in_cut_start_mat), "The length of in_cut_start_mat must equal to sequence length"
        if hid is None:
            hid = torch.zeros(x.shape[0], self.hidden_size)  # 默认初始化成0
        self.grucell.q_one = self.q_one_list[0]['forward']
        self.grucell.in_cut_start_list = self.in_cut_start_mat[0]['forward']
        self.grucell.lut_list = self.lut_mat[0]['forward']
        self.grucell_back.q_one = self.q_one_list[0]['backward']
        self.grucell_back.in_cut_start_list = self.in_cut_start_mat[0]['backward']
        self.grucell_back.lut_list = self.lut_mat[0]['backward']
        next_hid = self.grucell(x[:, 0, :], hid)
        next_hid_back = self.grucell_back(x[:, -1, :], hid)
        self.gru_summary[0] = {}
        self.gru_summary[0]['forward'] = copy.deepcopy(self.grucell.grucell_summary)
        self.gru_summary[0]['backward'] = copy.deepcopy(self.grucell_back.grucell_summary)
        if x.shape[1] == 1:
            return torch.cat((next_hid.detach(), next_hid_back.detach()), dim=1)
        else:
            for i in range(1, x.shape[1]):
                self.grucell.q_one = self.q_one_list[i]['forward']
                self.grucell.in_cut_start_list = self.in_cut_start_mat[i]['forward']
                self.grucell.lut_list = self.lut_mat[i]['forward']
                self.grucell_back.q_one = self.q_one_list[i]['backward']
                self.grucell_back.in_cut_start_list = self.in_cut_start_mat[i]['backward']
                self.grucell_back.lut_list = self.lut_mat[i]['backward']
                next_hid = self.grucell(x[:, i, :], next_hid)
                next_hid_back = self.grucell_back(x[:, -i - 1, :], next_hid_back)
                self.gru_summary[i] = {}
                self.gru_summary[i]['forward'] = copy.deepcopy(self.grucell.grucell_summary)
                self.gru_summary[i]['backward'] = copy.deepcopy(self.grucell_back.grucell_summary)
            return torch.cat((next_hid.detach(), next_hid_back.detach()), dim=1)


if __name__ == '__main__':
    input_size = 16
    hidden_size = 128
    x = torch.round(torch.randn(1, 37, input_size) * 128)  # batch size, sequence length, input size
    ref = nn.GRU(input_size, hidden_size, 1, batch_first=True, bidirectional=True)

    in_cut_start_mat = [{'forward':
                        {'GRU3_cut': 3,
                         'GRU6_cut': 3,
                         'GRU8_cut': 3,
                         'GRU10_cut': 3,
                         'GRU11_cut': 3,
                         'next_hid_cut': 3},
                         'backward':
                        {'GRU3_cut': 3,
                         'GRU6_cut': 3,
                         'GRU8_cut': 3,
                         'GRU10_cut': 3,
                         'GRU11_cut': 3,
                         'next_hid_cut': 3}}                         
                         ] * 37
    q_one_list = [{'forward': 127, 'backward': 127}] * 37
    lut_mat = [{'forward':
               {'sigmoid_r_d': 128,
                'sigmoid_r_m': 128,
                'sigmoid_z_d': 128,
                'sigmoid_z_m': 128,
                'tanh_d': 128,
                'tanh_m': 128},
                'backward':
               {'sigmoid_r_d': 128,
                'sigmoid_r_m': 128,
                'sigmoid_z_d': 128,
                'sigmoid_z_m': 128,
                'tanh_d': 128,
                'tanh_m': 128}}
                ] * 37

    model = GRU(input_size, hidden_size, in_cut_start_mat, q_one_list, lut_mat)
    ref.weight_ih_l0.data.clamp_(-1, 1).mul_(128).round_()
    ref.bias_ih_l0.data.clamp_(-1, 1).mul_(16384).round_()
    ref.weight_hh_l0.data.clamp_(-1, 1).mul_(128).round_()
    ref.bias_hh_l0.data.clamp_(-1, 1).mul_(16384).round_()
    ref.weight_ih_l0_reverse.data.clamp_(-1, 1).mul_(128).round_()
    ref.bias_ih_l0_reverse.data.clamp_(-1, 1).mul_(16384).round_()
    ref.weight_hh_l0_reverse.data.clamp_(-1, 1).mul_(128).round_()
    ref.bias_hh_l0_reverse.data.clamp_(-1, 1).mul_(16384).round_()
    model.grucell.in2hid_w[0].data = ref.weight_ih_l0[0:hidden_size, :].t()
    model.grucell.in2hid_w[1].data = ref.weight_ih_l0[hidden_size:2 * hidden_size, :].t()
    model.grucell.in2hid_w[2].data = ref.weight_ih_l0[2 * hidden_size:3 * hidden_size, :].t()
    model.grucell.in2hid_b[0].data = ref.bias_ih_l0[0:hidden_size]
    model.grucell.in2hid_b[1].data = ref.bias_ih_l0[hidden_size:2 * hidden_size]
    model.grucell.in2hid_b[2].data = ref.bias_ih_l0[2 * hidden_size:3 * hidden_size]
    model.grucell.hid2hid_w[0].data = ref.weight_hh_l0[0:hidden_size, :].t()
    model.grucell.hid2hid_w[1].data = ref.weight_hh_l0[hidden_size:2 * hidden_size, :].t()
    model.grucell.hid2hid_w[2].data = ref.weight_hh_l0[2 * hidden_size:3 * hidden_size, :].t()
    model.grucell.hid2hid_b[0].data = ref.bias_hh_l0[0:hidden_size]
    model.grucell.hid2hid_b[1].data = ref.bias_hh_l0[hidden_size:2 * hidden_size]
    model.grucell.hid2hid_b[2].data = ref.bias_hh_l0[2 * hidden_size:3 * hidden_size]

    model.grucell_back.in2hid_w[0].data = ref.weight_ih_l0_reverse[0:hidden_size, :].t()
    model.grucell_back.in2hid_w[1].data = ref.weight_ih_l0_reverse[hidden_size:2 * hidden_size, :].t()
    model.grucell_back.in2hid_w[2].data = ref.weight_ih_l0_reverse[2 * hidden_size:3 * hidden_size, :].t()
    model.grucell_back.in2hid_b[0].data = ref.bias_ih_l0_reverse[0:hidden_size]
    model.grucell_back.in2hid_b[1].data = ref.bias_ih_l0_reverse[hidden_size:2 * hidden_size]
    model.grucell_back.in2hid_b[2].data = ref.bias_ih_l0_reverse[2 * hidden_size:3 * hidden_size]
    model.grucell_back.hid2hid_w[0].data = ref.weight_hh_l0_reverse[0:hidden_size, :].t()
    model.grucell_back.hid2hid_w[1].data = ref.weight_hh_l0_reverse[hidden_size:2 * hidden_size, :].t()
    model.grucell_back.hid2hid_w[2].data = ref.weight_hh_l0_reverse[2 * hidden_size:3 * hidden_size, :].t()
    model.grucell_back.hid2hid_b[0].data = ref.bias_hh_l0_reverse[0:hidden_size]
    model.grucell_back.hid2hid_b[1].data = ref.bias_hh_l0_reverse[hidden_size:2 * hidden_size]
    model.grucell_back.hid2hid_b[2].data = ref.bias_hh_l0_reverse[2 * hidden_size:3 * hidden_size]

    y = model(x)
    print(y)
    # output, hidden = ref(x)
    # print(torch.sum(y - hidden.data))

        
