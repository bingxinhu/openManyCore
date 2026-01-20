import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.getcwd())
from generator.SoundSNN.snn_config import SNNConfig


class LIFactFun(nn.Module):
    def __init__(self):
        super(LIFactFun, self).__init__()

    def forward(self, input, thresh=64):
        fire = input.gt(thresh).float()
        return fire
            

class LIFFCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.thresh = config['thresh']
        self.beta = config['beta']
        self.decay = config['decay'] / 255
        self.hidden_size = hidden_size
        self.kernel = nn.Linear(input_size, hidden_size)
        self.act = LIFactFun()

    def forward(self, input_data, summary):
        self.batch_size = input_data.size(0)
        self.time_windows = input_data.size(1)
        synaptic_input = torch.zeros(self.batch_size, self.time_windows, self.hidden_size)
        summary['mlp'] = {}
        for time in range(self.time_windows):
            synaptic_input[:, time, :] = self.kernel(input_data[:, time, :])
            summary['mlp'][time] = synaptic_input[:, time, :].squeeze(0).detach()
        output = self.mem_update(synaptic_input, summary)
        return output
    
    # 输入突触连接为fc层 输入维度为[B, T, N]
    def mem_update(self, x, summary):
        output = self.__3Dmem_update(x, summary)
        return output
    
    # LIF神经元膜电位迭代部分
    def __3Dmem_update(self, x, summary):
        mem = torch.zeros_like(x[:, 0, :])
        outputs = torch.zeros_like(x)
        summary['mem'] = {}
        for t in range(self.time_windows): 
            mem = mem + x[:, t, :]
            spike = self.act(mem.clone(), self.thresh)
            output = self.act(mem.clone(), self.thresh)
            outputs[:, t, :] = output
            mem = mem * (1 - spike.detach())
            mem = mem * self.decay + self.beta
            summary['mem'][t] = mem.squeeze(0).detach()
        return outputs


class SNN(nn.Module):
    def __init__(self, config):
        super(SNN, self).__init__()
        
        self.fc1 = LIFFCNeuron(16, 128, config['fc1'])
        self.fc2 = LIFFCNeuron(128, 128, config['fc2'])
        self.fc3 = LIFFCNeuron(128, 128, config['fc3'])
        self.fc4 = LIFFCNeuron(128, 1, config['fc4'])

        self.summary = {}

    def forward(self, x):
        self.summary['fc1'] = {}
        self.summary['fc1']['input'] = x.squeeze(0).detach()
        self.summary['fc1']['weight'] = self.fc1.kernel.weight.data.detach()
        self.summary['fc1']['bias'] = self.fc1.kernel.bias.data.detach()
        x = self.fc1(x, self.summary['fc1'])
        self.summary['fc1']['output'] = x.squeeze(0).detach()

        self.summary['fc2'] = {}
        self.summary['fc2']['input'] = x.squeeze(0).detach()
        self.summary['fc2']['weight'] = self.fc2.kernel.weight.data.detach()
        self.summary['fc2']['bias'] = self.fc2.kernel.bias.data.detach()
        x = self.fc2(x, self.summary['fc2'])
        self.summary['fc2']['output'] = x.squeeze(0).detach()

        self.summary['fc3'] = {}
        self.summary['fc3']['input'] = x.squeeze(0).detach()
        self.summary['fc3']['weight'] = self.fc3.kernel.weight.data.detach()
        self.summary['fc3']['bias'] = self.fc3.kernel.bias.data.detach()        
        x = self.fc3(x, self.summary['fc3'])
        self.summary['fc3']['output'] = x.squeeze(0).detach()

        self.summary['fc4'] = {}
        self.summary['fc4']['input'] = x.squeeze(0).detach()
        self.summary['fc4']['weight'] = self.fc4.kernel.weight.data.detach()
        self.summary['fc4']['bias'] = self.fc4.kernel.bias.data.detach()
        x = self.fc4(x, self.summary['fc4'])
        self.summary['fc4']['output'] = x.squeeze(0).detach()

        return torch.mean(x, dim=1)


if __name__ == '__main__':
    sequence_length = 39
    x = torch.randn((1, sequence_length, 16))
    x = x.mul(128).round().clamp(-128, 127)

    snn_config = SNNConfig()
    model = SNN(config=snn_config)
    for name, module in model.named_modules():    
        if isinstance(module, LIFFCNeuron):
            module.kernel.weight.data = module.kernel.weight.data.mul(128).round().clamp(-128, 127)
            module.kernel.bias.data = module.kernel.bias.data.mul(128).round().clamp(-2**31, 2**31 - 1)

    y = model(x)
    print(model.summary['fc1']['mlp'][0])
    print(y)