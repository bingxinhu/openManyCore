import torch

def get_core_id(x: int, y: int) -> int:
    if y % 2 == 0:
        return x + y * 16
    else:
        return (y + 1) * 16 - x - 1

def fuse2d_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    assert (conv.training == bn.training), \
        "Conv and BN both must be in the same mode (train or eval)."
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels,
                               kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
                               dilation=conv.dilation, groups=conv.groups, bias=True, padding_mode=conv.padding_mode)

    if bn.affine:
        gamma = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        new_conv.weight.data = conv.weight.data * gamma.view(-1, 1, 1, 1)
        if conv.bias is not None:
            new_conv.bias.data = gamma * conv.bias.data - gamma * bn.running_mean + bn.bias.data
        else:
            new_conv.bias.data = bn.bias.data - gamma * bn.running_mean
    else:
        "affine 为 False 的情况，gamma=1, beta=0"
        gamma = 1 / torch.sqrt(bn.running_var + bn.eps)
        new_conv.weight.data = conv.weight.data * gamma
        if conv.bias is not None:
            new_conv.bias.data = gamma * conv.bias.data - gamma * bn.running_mean
        else:
            new_conv.bias.data = - gamma * bn.running_mean
    return new_conv
