import torch
import torch.nn as nn
import torch.nn.functional as F

lens = 0.5
v_th = 0.


class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):
        """
        result = (x >> (2 * in_cut_start).clip()
        type_out: 0-int32, 1-int8
        """
        super(Cut, self).__init__()
        self.en = en
        self.in_cut_start = in_cut_start
        self.type_out = type_out
        if type_out == 0:
            self.max, self.min = 0x7fffffff, -0x80000000
        elif type_out == 1:
            self.max, self.min = 0x7f, -0x80
        else:
            raise ValueError

    def forward(self, x):
        if self.en:
            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)
        return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__, self.en,
                                                                    self.in_cut_start, self.type_out)


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


class main_net(nn.Module):
    def __init__(self, state_num, input_num, hidden_num, quantization=True, in_cut_start=None):
        super(main_net, self).__init__()
        self.linear_T_T = nn.Linear(input_num, hidden_num, bias=False)
        self.linear_S_T = nn.Linear(state_num, hidden_num, bias=False)
        self.cut_tt = Cut(en=quantization, in_cut_start=in_cut_start['tt'])
        self.cut_st = Cut(en=quantization, in_cut_start=in_cut_start['st'])
        self.cut_tt_st = Cut(en=quantization, in_cut_start=in_cut_start['tt_st'])
        self.linear_T_S = nn.Linear(hidden_num, state_num, bias=False)
        self.linear_S_S = nn.Linear(state_num, state_num, bias=False)
        self.summary = {
            'linear_tt': {},
            'linear_st': {},
            'linear_ts': {},
            'linear_ss': {},
        }

    def forward(self, init_state, inputs):
        self.summary['linear_tt']['weight'] = self.linear_T_T.weight.data.clone().detach()
        self.summary['linear_st']['weight'] = self.linear_S_T.weight.data.clone().detach()
        self.summary['linear_ts']['weight'] = self.linear_T_S.weight.data.clone().detach()
        self.summary['linear_ss']['weight'] = self.linear_S_S.weight.data.clone().detach()

        state = init_state
        state_list = [state]

        self.summary['init_state'] = init_state.clone().detach()
        self.summary['inputs'] = inputs.clone().detach()

        t1 = self.linear_T_T(inputs)
        t2 = self.linear_S_T(state)

        t1 = self.cut_tt(t1)
        t2 = self.cut_st(t2)

        self.summary['t1_cut'] = t1.clone().detach()
        self.summary['t2_cut'] = t2.clone().detach()

        hidden = t1 * t2

        self.summary['hidden_1'] = hidden.clone().detach()

        hidden = self.cut_tt_st(hidden)

        self.summary['hidden_1_cut'] = hidden.clone().detach()

        t3 = self.linear_T_S(hidden)
        t4 = self.linear_S_S(state)

        self.summary['t3'] = t3.clone().detach()
        self.summary['t4'] = t4.clone().detach()

        hidden = t3 + t4

        self.summary['hidden_2'] = hidden.clone().detach()

        state = act_fun(hidden)

        self.summary['act_fun'] = state.clone().detach()

        state_list.append(state)
        state = torch.stack(state_list, dim=0)

        state = state.permute(1, 0, 2)  # 1, 2, 5

        self.summary['output'] = state.clone().detach()
        return state


if __name__ == '__main__':
    net = main_net(state_num=5, input_num=4, hidden_num=100, in_cut_start={'tt_st': 4})
    s = torch.rand((1, 5))
    t = torch.rand((1, 4))
    y = net(s, t)
    print(net)
