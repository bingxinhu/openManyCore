import torch
import os
import sys

sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from GRU import GRU, cut, GRUCell

input_channels = 8
output_channels = 16
hidden_size = 128


class NeuralNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size,
                 in_cut_start_mat, q_one_list, lut_mat, quantization_en=True, conv1_cut=0, mlp_cut=0,
                 mlp_tanh_d=128, mlp_tanh_m=128):
        super(NeuralNetwork, self).__init__()

        self.conv2d = torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(3, 3),
                                      stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #   self.conv2d_2=torch.nn.Conv2d(in_channels=hid_channels, out_channels=output_channels, kernel_size=(3,3),
        #                               stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

        self.RelU = nn.ReLU()

        self.maxSeq = nn.Sequential()
        self.maxSeq.add_module('maxpool1', torch.nn.MaxPool2d(kernel_size=(8, 1), stride=(7, 1), padding=0, dilation=1))
        self.maxSeq.add_module('maxpool2', torch.nn.MaxPool2d(kernel_size=(8, 1), stride=(7, 1), padding=0, dilation=1))
        self.maxSeq.add_module('maxpool3', torch.nn.MaxPool2d(kernel_size=(5, 1), stride=(1, 1), padding=0, dilation=1))

        #   self.BatchNorm_2 = nn.BatchNorm2d(num_features=output_channels, affine=True)

        self.myGRU = GRU(input_num=output_channels, hidden_num=hidden_size,
                         in_cut_start_mat=in_cut_start_mat, q_one_list=q_one_list,
                         lut_mat=lut_mat, quantization_en=True)

        self.MLP = nn.Linear(in_features=2 * hidden_size, out_features=2, bias=True)

        self.tanh = nn.Tanh()

        self.quantization_en = quantization_en
        self.conv1_cut = conv1_cut
        self.mlp_cut = mlp_cut
        self.mlp_tanh_d = mlp_tanh_d
        self.mlp_tanh_m = mlp_tanh_m

        self.summary = dict()
        self.summary['gru'] = self.myGRU.gru_summary
        self.summary['conv1'] = {}
        self.summary['relu1'] = {}
        self.summary['maxpool'] = {}
        self.summary['mlp'] = {}
        self.summary['mlp_tanh'] = {}

    def forward(self, x):
        batch_size = x.size(0)
        self.summary['conv1']['input'] = x.squeeze(0).detach()
        x = self.conv2d(x)
        self.summary['conv1']['output'] = x.squeeze(0).detach()
        self.summary['conv1']['weight'] = self.conv2d.weight.data
        self.summary['conv1']['bias'] = self.conv2d.bias.data
        x = cut(x, in_cut_start=self.conv1_cut, en=self.quantization_en)
        self.summary['conv1']['output_cut'] = x.squeeze(0).detach()

        self.summary['relu1']['input'] = x.squeeze(0).detach()
        x = self.RelU(x)
        self.summary['relu1']['output'] = x.squeeze(0).detach()

        self.summary['maxpool']['input'] = x.squeeze(0).detach()
        # x.size torch.Size([16, 3, 255, 39])
        x = self.maxSeq(x)  # x.size torch.Size([N, 16, 1, 39])
        self.summary['maxpool']['output'] = x.squeeze(0).detach()
        x = x.view(batch_size, output_channels, -1).permute(0, 2, 1)  # x.size torch.Size([N, 16, 1, 39]) -> [N, 39, 16]
        
        x = self.myGRU(x)

        self.summary['mlp']['input'] = x.squeeze(0).detach()
        x = self.MLP(x)
        self.summary['mlp']['output'] = x.squeeze(0).detach()
        self.summary['mlp']['weight'] = self.MLP.weight.data
        self.summary['mlp']['bias'] = self.MLP.bias.data
        x = cut(x, in_cut_start=self.mlp_cut, en=self.quantization_en)
        self.summary['mlp']['output_cut'] = x.squeeze(0).detach()

        self.summary['mlp_tanh']['input'] = x.squeeze(0).detach()
        x = x / self.mlp_tanh_d
        x = self.tanh(x)
        x = torch.floor(x * self.mlp_tanh_m)
        self.summary['mlp_tanh']['output'] = x.squeeze(0).detach()

        return x


if __name__ == '__main__':
    from quantization_config import QuantizationConfig
    from sound_tracking_fp32 import ReferenceModel
    from sound_tracking_fp32 import snn_n_model
    from utils import fuse2d_conv_bn

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reference = ReferenceModel(input_channels, output_channels, hidden_size)
    # snn = snn_n_model()
    # reference.snn = snn.to(device)
    state_dict = torch.load('sound_tracking.pth', map_location=device)
    state_dict.pop('snn.fc1.kernel.weight')
    state_dict.pop('snn.fc1.kernel.bias')
    state_dict.pop('snn.fc2.kernel.weight')
    state_dict.pop('snn.fc2.kernel.bias')
    state_dict.pop('snn.fc3.kernel.weight')
    state_dict.pop('snn.fc3.kernel.bias')
    state_dict.pop('snn.fc5.kernel.weight')
    state_dict.pop('snn.fc5.kernel.bias')
    reference.load_state_dict(state_dict)

    # Conv BN融合
    fused_conv2d = fuse2d_conv_bn(reference.conv2d, reference.BatchNorm)

    x = 0.05 * torch.randn((1, 8, 257, 41)).mul(128).floor()  # 这里需要用真实数据
    para = QuantizationConfig(sequence_length=39)
    model = NeuralNetwork(input_channels, output_channels, hidden_size,
                          in_cut_start_mat=para['in_cut_start'], q_one_list=para['q_one'],
                          lut_mat=para['lut'], quantization_en=True, conv1_cut=para['conv1'], mlp_cut=para['mlp'],
                          mlp_tanh_d=para['mlp_tanh_d'], mlp_tanh_m=para['mlp_tanh_m'])

    # 预训练参数加载
    model.conv2d.weight.data = fused_conv2d.weight.data
    model.conv2d.bias.data = fused_conv2d.bias.data
    model.MLP.weight.data = reference.MLP.weight.data
    model.MLP.bias.data = reference.MLP.bias.data
    model.myGRU.grucell.in2hid_w[0].data = reference.myGRU.gru.grucell.in2hid_w[0].data
    model.myGRU.grucell.in2hid_w[1].data = reference.myGRU.gru.grucell.in2hid_w[1].data
    model.myGRU.grucell.in2hid_w[2].data = reference.myGRU.gru.grucell.in2hid_w[2].data
    model.myGRU.grucell.in2hid_b[0].data = reference.myGRU.gru.grucell.in2hid_b[0].data
    model.myGRU.grucell.in2hid_b[1].data = reference.myGRU.gru.grucell.in2hid_b[1].data
    model.myGRU.grucell.in2hid_b[2].data = reference.myGRU.gru.grucell.in2hid_b[2].data
    model.myGRU.grucell.hid2hid_w[0].data = reference.myGRU.gru.grucell.hid2hid_w[0].data
    model.myGRU.grucell.hid2hid_w[1].data = reference.myGRU.gru.grucell.hid2hid_w[1].data
    model.myGRU.grucell.hid2hid_w[2].data = reference.myGRU.gru.grucell.hid2hid_w[2].data
    model.myGRU.grucell.hid2hid_b[0].data = reference.myGRU.gru.grucell.hid2hid_b[0].data
    model.myGRU.grucell.hid2hid_b[1].data = reference.myGRU.gru.grucell.hid2hid_b[1].data
    model.myGRU.grucell.hid2hid_b[2].data = reference.myGRU.gru.grucell.hid2hid_b[2].data
    model.myGRU.grucell_back.in2hid_w[0].data = reference.myGRU.gru_bkwd.grucell.in2hid_w[0].data
    model.myGRU.grucell_back.in2hid_w[1].data = reference.myGRU.gru_bkwd.grucell.in2hid_w[1].data
    model.myGRU.grucell_back.in2hid_w[2].data = reference.myGRU.gru_bkwd.grucell.in2hid_w[2].data
    model.myGRU.grucell_back.in2hid_b[0].data = reference.myGRU.gru_bkwd.grucell.in2hid_b[0].data
    model.myGRU.grucell_back.in2hid_b[1].data = reference.myGRU.gru_bkwd.grucell.in2hid_b[1].data
    model.myGRU.grucell_back.in2hid_b[2].data = reference.myGRU.gru_bkwd.grucell.in2hid_b[2].data
    model.myGRU.grucell_back.hid2hid_w[0].data = reference.myGRU.gru_bkwd.grucell.hid2hid_w[0].data
    model.myGRU.grucell_back.hid2hid_w[1].data = reference.myGRU.gru_bkwd.grucell.hid2hid_w[1].data
    model.myGRU.grucell_back.hid2hid_w[2].data = reference.myGRU.gru_bkwd.grucell.hid2hid_w[2].data
    model.myGRU.grucell_back.hid2hid_b[0].data = reference.myGRU.gru_bkwd.grucell.hid2hid_b[0].data
    model.myGRU.grucell_back.hid2hid_b[1].data = reference.myGRU.gru_bkwd.grucell.hid2hid_b[1].data
    model.myGRU.grucell_back.hid2hid_b[2].data = reference.myGRU.gru_bkwd.grucell.hid2hid_b[2].data

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.mul_(64).floor_()
            module.bias.data.mul_(64 * 128).floor_()
        elif isinstance(module, nn.Linear):
            module.weight.data.mul_(64).floor_()
            module.bias.data.mul_(64 * 128).floor_()
        elif isinstance(module, GRUCell):
            module.in2hid_w[0].data.mul_(64).floor_()
            module.in2hid_w[1].data.mul_(64).floor_()
            module.in2hid_w[2].data.mul_(64).floor_()
            module.in2hid_b[0].data.mul_(64 * 128).floor_()
            module.in2hid_b[1].data.mul_(64 * 128).floor_()
            module.in2hid_b[2].data.mul_(64 * 128).floor_()
            module.hid2hid_w[0].data.mul_(64).floor_()
            module.hid2hid_w[1].data.mul_(64).floor_()
            module.hid2hid_w[2].data.mul_(64).floor_()
            module.hid2hid_b[0].data.mul_(64 * 128).floor_()
            module.hid2hid_b[1].data.mul_(64 * 128).floor_()
            module.hid2hid_b[2].data.mul_(64 * 128).floor_()
    
    y = model(x)
    print(y)
