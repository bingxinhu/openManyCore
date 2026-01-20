import torch
import torch.nn as nn
from math import sqrt

class GRUCell(nn.Module):
    """自定义GRUCell"""
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置) 
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
        lb, ub = -sqrt(1/hidden_size), sqrt(1/hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    def forward(self, x, hid):
        r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
                       torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
        next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
        return next_hid


class GRU(nn.Module):   
    def __init__(self, input_num, hidden_num):
        super(GRU, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = GRUCell(input_num, hidden_num)
        self.grucell_back = GRUCell(input_num, hidden_num)

    def forward(self, x, hid=None):
        if hid is None:
            hid = torch.zeros(x.shape[0], self.hidden_size)  # 默认初始化成0
        next_hid = self.grucell(x[:, 0, :], hid)
        next_hid_back = self.grucell_back(x[:, -1, :], hid)
        next_hid_stack = torch.stack((next_hid, next_hid_back))
        if x.shape[1] == 1:
            return next_hid_stack.detach()
        else:
            for i in range(1, x.shape[1]):
                next_hid = self.grucell(x[:, i, :], next_hid)
                next_hid_back = self.grucell_back(x[:, -i - 1, :], next_hid_back)
            next_hid_stack = torch.stack((next_hid, next_hid_back))
            return next_hid_stack.detach()


if __name__ == '__main__':
    input_size = 4
    hidden_size = 5
    x = torch.randn(1, 3, input_size)
    ref = nn.GRU(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
    # for p in ref.named_parameters():
    #     print(p[0])
    model = GRU(input_size, hidden_size)
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
    # for p in ref.named_parameters():
    #     print(p)
    output, hidden = ref(x)
    print(hidden)

        
