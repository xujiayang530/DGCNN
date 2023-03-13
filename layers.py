import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        # self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())   #Pytorch nn.Parameter() 创建模型可训练参数  https://blog.csdn.net/hxxjxw/article/details/107904012
                                                                                 #torch.FloatTensor(a,b)   随机生成aXb格式的tensor

        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out))
        nn.init.xavier_normal_(self.weight)  #大致就是使可训练参数服从正态分布
        self.bias = None
        if bias:
            # self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            self.bias = nn.Parameter(torch.FloatTensor(num_out))
            nn.init.zeros_(self.bias)                 #有偏置 则置为0

    def forward(self, x, adj):
        out = torch.matmul(adj, x)    #矩阵/向量 乘法       
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
