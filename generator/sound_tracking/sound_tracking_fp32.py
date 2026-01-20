import torch
import torch.nn as nn
from math import sqrt

C = 4
input_channels = 2*C
hid_channels = 16
output_channels = 16
frequency_num = 513
hidden_size = 128
num_layers = 1
learning_rate =  1e-3
batch_size = 256


#######################################################
# cells 
# LIF激活函数
# 推理时只需要gt部分
#######################################################
class LIFactFun(torch.autograd.Function):
    lens = 0.5      # LIF激活函数的梯度近似参数，越小则梯度激活区域越窄                       
    sigma = 1       # 高斯梯度近似时的sigma
    use_rect_approx = True # 选择方梯度近似方法【Switch Flag】
    use_gause_approx = False # 选择高斯梯度近似方法【Switch Flag】
    def __init__(self):
        super(LIFactFun, self).__init__()

    # 阈值激活，带有输入阈值，阈值可训练
    @staticmethod
    def forward(ctx, input, thresh=0.5):
        fire = input.gt(thresh).float() 
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return fire 

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = 0
        thresh = ctx.thresh
        if LIFactFun.use_rect_approx:
            temp = abs(input - thresh) < LIFactFun.lens  
            grad = grad_input*temp.float()/(2*LIFactFun.lens)
        elif LIFactFun.use_gause_approx:
            temp = 0.3989422804014327 / LIFactFun.sigma*torch.exp(-0.5/(LIFactFun.sigma**2)*(input-thresh+LIFactFun.bias)**2) 
            grad = grad_input * temp.float()
        return grad, None

            
#######################################################
# cells 
# LIF神经元
# 所有参数都可训练的LIF神经元
# 阈值按ARLIF的训练方法
# decay使用sigmoid归一化
# FC处理2维输入，Conv处理4维输入
# update: 2021-08-25
# author: Linyh
#######################################################
class LIFFCNeuron(nn.Module):

    def __init__(self,inputSize,hiddenSize):
        self.device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.thresh =  0.5
        self.beta = 0
        self.decay =torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=False).to(self.device)  
        self.hiddenSize = hiddenSize
        self.kernel = nn.Linear(inputSize, hiddenSize)   

    def forward(self,input_data):
        self.batchSize = input_data.size(0)#adaptive for mul-gpu training
        self.timeWindows = input_data.size(1)
        synaptic_input = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device)
        for time in range(self.timeWindows):
            synaptic_input[:,time,:] = self.kernel(input_data[:,time,:]).clamp_(-1.0, 1.0)
        output = self.mem_update(synaptic_input)
        return output
    
    # 输入突触连接为fc层 输入维度为[B,T,N]
    def mem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        dim = len(x.size())
        output = self.__3Dmem_update(x,init_mem,spikeAct)      
        output = output.clamp_(0.0, 1.0)
        return output
    
    #######################################################
    # LIF神经元膜电位迭代部分
    #######################################################
    def __3Dmem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):

        time_window = x.size(1)
        mem = torch.zeros_like(x[:,0,:]).to(self.device)  
        outputs = torch.zeros_like(x).to(self.device)  
        if init_mem is not None:
            mem = init_v
        for t in range(time_window):
            
            mem = mem + x[:,t,:].clamp_(-1.0, 1.0) #超过1也没关系？
            spike =  spikeAct(mem.clone(),self.thresh) #0~1
            output = spikeAct(mem.clone(),self.thresh)  #0~1
            outputs[:,t,:] = output #0~1
            mem = mem*(1 - spike.detach()) #0~1
            mem = (mem * self.decay + self.beta).clamp_(-1.0, 1.0)  #0~1
            
        return outputs

#######################################################
# SNN part
#######################################################
class snn_n_model(nn.Module):
    def __init__(self, layer_size=None):
        super(snn_n_model, self).__init__()
        
        self.fc1 = LIFFCNeuron(16,128)
        self.fc2 = LIFFCNeuron(128,128)
        self.fc3 = LIFFCNeuron(128,128)
        #self.fc4 = LIFFCNeuron(128,128)
        self.fc5 = LIFFCNeuron(128,1)
        

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
        x = self.fc5(x)
       
        return torch.mean(x,dim=1)


#######################################################
# 自定义GRUCell
#######################################################
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
        
        xmm =   (torch.mm(x, self.in2hid_w[0])+ self.in2hid_b[0]).clamp_(-1.0, 1.0)
        hidmm = (torch.mm(hid, self.hid2hid_w[0])+ self.hid2hid_b[0]).clamp_(-1.0, 1.0)
        r = (xmm + hidmm).clamp_(-1.0, 1.0)#.clamp_(-1.0, 1.0)
        r = torch.sigmoid(r) 
        
        xmm2 =   (torch.mm(x, self.in2hid_w[1])+ self.in2hid_b[1]).clamp_(-1.0, 1.0)
        hidmm2 = (torch.mm(hid, self.hid2hid_w[1])+ self.hid2hid_b[1]).clamp_(-1.0, 1.0)
        z = (xmm2 + hidmm2).clamp_(-1.0, 1.0).clamp_(-1.0, 1.0)
        z = torch.sigmoid(z)
        
        xmm3 =   (torch.mm(x, self.in2hid_w[2])+ self.in2hid_b[2]).clamp_(-1.0, 1.0)
        hidmm3 = (torch.mm(hid, self.hid2hid_w[2])+ self.hid2hid_b[2]).clamp_(-1.0, 1.0)
        hidmm_r = torch.mul(r,hidmm3).clamp_(-1.0, 1.0)
        n = (xmm3 + hidmm_r).clamp_(-1.0, 1.0) 
        n = torch.tanh(n)
        
        next_hid = torch.mul((1 - z), n).clamp_(-1.0, 1.0) + torch.mul(z, hid).clamp_(-1.0, 1.0)
        return next_hid.clamp_(-1.0, 1.0)

class GRUModel(nn.Module):   
    def __init__(self, input_num, hidden_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = GRUCell(input_num, hidden_num)

    def forward(self, x, hid):
        next_hid = self.grucell(x, hid)  
        return next_hid


class myGRU(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(myGRU, self).__init__()
        self.gru = GRUModel(input_num, hidden_num)
        self.gru_bkwd = GRUModel(input_num, hidden_num)
        self.hidden_size = hidden_num
        
    def forward(self, x, hid=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if hid is None:
            hid = torch.zeros(x.size(0), self.hidden_size, device=device)  
            hid_bkwd = torch.zeros(x.size(0), self.hidden_size, device=device) 
            
        for i in range(0, x.size(1)):
            hid = self.gru(x[:, i, :], hid)
            
        for i in range(1, x.size(1)+1):
            hid_bkwd = self.gru_bkwd(x[:, x.size(1)-i, :], hid_bkwd)   

        output = torch.cat([hid,hid_bkwd],dim=1)
        return output
    


#######################################################
# 总网络
#######################################################
class ReferenceModel(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size):
        super(ReferenceModel, self).__init__()
        self.conv2d=torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(3,3), 
                                    stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')    
        self.maxSeq = nn.Sequential()
        self.maxSeq.add_module('maxpool1',torch.nn.MaxPool2d(kernel_size=(8,1), stride=(7,1), padding=0, dilation=1))
        self.maxSeq.add_module('maxpool2',torch.nn.MaxPool2d(kernel_size=(8,1), stride=(7,1), padding=0, dilation=1))
        self.maxSeq.add_module('maxpool3',torch.nn.MaxPool2d(kernel_size=(5,1), stride=(1,1), padding=0, dilation=1))

        self.RelU = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d(num_features=output_channels, affine=True)

        self.myGRU = myGRU(input_num=output_channels,hidden_num = hidden_size)
        
        self.MLP = nn.Linear(in_features = 2*hidden_size, out_features = 2, bias=True)
        
        self.tanh=nn.Tanh()
        
        self.snn = None
        
    def forward(self, x):

        batch_size=x.size(0)
        x = self.conv2d(x).clamp_(-1.0, 1.0)
        x = self.BatchNorm(x).clamp_(-1.0, 1.0) 
        x = self.RelU(x)
        x = self.maxSeq(x)                         
        x = x.view(batch_size,output_channels,-1).permute(0,2,1) 
        
        cls = self.snn(x)#截断梯度，可以去掉以选择同时训练
        
        x = self.myGRU(x)
        x = x.clamp_(-1.0, 1.0) 
        x = self.MLP(x)
        x = x.clamp_(-1.0, 1.0)
        angle = self.tanh(x)
        
        return cls,angle
    
    
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, y):
        loss = self.loss(pred,y)
        return loss
        

snn = snn_n_model()  


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
        