#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
#-------------------------------------------------------------------------------
# Preparation of data and helper functions.
#-------------------------------------------------------------------------------
import io
import os
import math
import tarfile
import multiprocessing

import scipy
import requests
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from utils import *

print(torch.__version__)
print(torchaudio.__version__)
# torch.cuda.set_device(6)
device = torch.device("cpu")
print(device)


# # 数据预处理

# In[2]:


import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

class CustomSoundDataset(Dataset):
    def __init__(self, labels_dir, sound_dir, transform=None, target_transform=None):
        self.labels_dir = pd.read_csv(labels_dir)
        self.sound_dir = sound_dir
        self.transform = transform
        self.target_transform = target_transform             

    def __len__(self):
        return len(self.labels_dir)

    def __getitem__(self, idx):
        sound_path = os.path.join(self.sound_dir, self.labels_dir.iloc[idx, 0])
        #print(sound_path)
        sound, sample_rate = torchaudio.load(sound_path)
        label = self.labels_dir.iloc[idx, 1]
        
        if self.labels_dir.shape[1] > 2:
            cls = self.labels_dir.iloc[idx,2]
            cls = int(cls)
        else:
            cls = 1

        if self.transform:
            for trans in self.transform:
                sound = trans(sound)
        if self.target_transform:
            for single_transform in self.target_transform:
                label = single_transform(label)  
        label = torch.Tensor(label)
        cls = torch.Tensor([cls])
        sample = {"sound": sound, "label": label,"cls":cls}
        return sample


def strlist_convert(strlabel):
    strlabel = strlabel.strip('[').strip(']')
    strlabel = strlabel.split()
    flolabel=np.array(strlabel,dtype=np.float32)
    return flolabel

n_fft = 512
win_length = None
hop_length = 256

# define transformation
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=None
)

def double_spec(wave):
    wave_rel = wave[:,:,:,0] 
    wave_img = wave[:,:,:,1]
    wave = torch.cat((wave_rel,wave_img),0)
    return wave

print("done")


# # Case2.1 和 2.2 的完整模型代码

# In[3]:


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



class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.loss2 = nn.BCELoss()
    def forward(self, pred, y , cls, cls_label):
        loss = 0
        if cls_label[0] >= 0:
            loss += 0.1 * self.loss2(cls, cls_label)
            loss += self.loss(pred*cls_label,y*cls_label)
        else:
            loss += self.loss(pred,y)
            
        return loss
        
model = 0

if True:
    from quantization_config import QuantizationConfig
    from sound_tracking_fp32 import ReferenceModel
    from sound_tracking_fp32 import snn_n_model
    from utils import fuse2d_conv_bn

    device = torch.device('cpu')
    reference = ReferenceModel(input_channels, output_channels, hidden_size)
    # snn = snn_n_model()
    # reference.snn = snn.to(device)
    state_dict = torch.load('./sound_tracking.pth', map_location=device)
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
                          lut_mat=para['lut'], quantization_en=False, conv1_cut=para['conv1'], mlp_cut=para['mlp'],
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

#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             module.weight.data.mul_(16384 * 16384).floor_()
#             module.bias.data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#         elif isinstance(module, nn.Linear):
#             module.weight.data.mul_(16384 * 16384).floor_()
#             module.bias.data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#         elif isinstance(module, GRUCell):
#             module.in2hid_w[0].data.mul_(16384 * 16384).floor_()
#             module.in2hid_w[1].data.mul_(16384 * 16384).floor_()
#             module.in2hid_w[2].data.mul_(16384 * 16384).floor_()
#             module.in2hid_b[0].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#             module.in2hid_b[1].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#             module.in2hid_b[2].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#             module.hid2hid_w[0].data.mul_(16384 * 16384).floor_()
#             module.hid2hid_w[1].data.mul_(16384 * 16384).floor_()
#             module.hid2hid_w[2].data.mul_(16384 * 16384).floor_()
#             module.hid2hid_b[0].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#             module.hid2hid_b[1].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
#             module.hid2hid_b[2].data.mul_(16384 * 16384 * 16384 * 16384).floor_()
    


# # test Loop部分

# In[4]:


def test_loop(dataloader, model, loss_fn,writer,epoch,device):
    import matplotlib 
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import matplotlib.cm as cm
#     plt.figure(figsize=(10,10))
    import math
    model.eval()
    size = len(dataloader.dataset)
    test_loss= 0
    count=0
    count_angle = 0
    correct2 = 0
    correct = 0
    correct3 = 0
    for name,parameters in model.named_parameters():
        writer.add_histogram(name, parameters.detach().cpu().numpy(),epoch)
    with torch.no_grad():
        for sample in dataloader:
            x = sample['sound'].to(device).float()
            for i in range(x.size(0)):
                x[i,...] = x[i,...]/torch.max(abs(x[i,...]))
#             x = x.mul(16384 * 16384).round()
            y = sample['label'].float()
            cls_label = sample['cls'].float()
            
            pred = model(x)
            cls = torch.ones([pred.size(0),1])
            
            pred = pred.cpu().float()
            cls = cls.cpu().float()
            loss = loss_fn(pred, y,cls,cls_label).float()
            test_loss += loss.item()

            batchsize = y.size(0)
            count += batchsize
            
            dot_product = pred[:,1]*y[:,1] + pred[:,0]*y[:,0]
            norm1 = (pred[:,1]**2+pred[:,0]**2)**0.5
            norm2 = (y[:,1]**2+y[:,0]**2)**0.5
            cos_theta = dot_product/(norm1*norm2)    
            theta = np.arccos(cos_theta.numpy())
            cls = cls > 0.5
            
#             print(theta)

            for i in range(batchsize):
                if cls_label[i] != 0:
                    count_angle +=1
                    if theta[i] < 3.1415926535/6:
                        correct += 1
                    if theta[i] < 3.1415926535/4:
                        correct2 += 1            
                if cls[i] == cls_label[i]:
                    correct3 += 1
                        
                        
#             colors = cm.rainbow(np.linspace(0, 1, batchsize))

#             for i in range(batchsize):
#                 c = colors[i]
#                 plt.scatter(pred[i,0]/norm1[i], pred[i,1]/norm1[i],c=c,marker = 'x')
#                 plt.scatter(y[i,0]/norm2[i], y[i,1]/norm2[i],c=c)
#                 x1 = np.linspace(pred[i,0]/norm1[i],y[i,0]/norm2[i],100)
#                 y1 = np.linspace(pred[i,1]/norm1[i],y[i,1]/norm2[i],100)
#                 plt.plot(x1, y1,c=c)
#                 if theta[i] < 3.1415926535/6:
#                     correct +=1
#                 if theta[i] < 3.1415926535/4:
#                     correct2 +=1   
#             plt.show()
                    
            
    test_loss /= count
    print(f"30 d Test: \n Accuracy: {(100*correct)/count_angle:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"45 d Test: \n Accuracy: {(100*correct2)/count_angle:>0.1f}% \n")
    print(f"clssisfication Accuracy: {(100*correct3)/count:>0.1f}% \n")
    writer.add_scalar('test_acc', 100*correct/count_angle, epoch)
    writer.add_scalar('test_acc2', 100*correct2/count_angle, epoch)
    writer.add_scalar('test_acc3', 100*correct3/count, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    
    return (100*correct)/count_angle
    
print("done")


# # main

# In[5]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
from tensorboardX import SummaryWriter
writer = SummaryWriter(comment = 'runs/with_cls')


torch.set_default_tensor_type(torch.FloatTensor)


model.to(device)

#for p in model.parameters(): #不训练
#    p.requires_grad=False
batch_size = 128
loss_fn = My_loss()
train_labels_dir = '../data/sound_dataset/train.csv'
test_labels_dir = '../data/sound_dataset/test.csv'
sound_dir = '../data/sound_dataset/'

training_data = CustomSoundDataset(train_labels_dir,sound_dir,
                                   transform=[spectrogram,double_spec],
                                   target_transform=[strlist_convert])
test_data = CustomSoundDataset(test_labels_dir,sound_dir,
                               transform=[spectrogram,double_spec],
                               target_transform=[strlist_convert])
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=batch_size, shuffle=True)

epochs = 1
best_acc = 0
iter_count = 0


for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    if e % 10 == 0:
        model.eval()
        acc = test_loop(test_dataloader, model, loss_fn,writer,e,device = device)
        if acc > best_acc:
            best_acc = acc

print("Done!")



# In[ ]:




