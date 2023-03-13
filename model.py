import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution, Linear
from utils import generate_cheby_adj, normalize_A


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()   #https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L):
        device = x.device
        adj = generate_cheby_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers
        #num_out: num_features_out
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  #对第二维（第一维为batch_size)进行标准化 
        self.fc1 = Linear(xdim[1] * num_out, 32)
        #self.fc2=Linear(64, 32)
        self.fc3=Linear(32, 8)
        self.fc4=Linear(8, nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)   #因为第三维 才为特征维度
        L = normalize_A(self.A)   #A是自己设置的59 * 59的可训练参数  及邻接矩阵
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        #result=F.relu(self.fc2(result))
        result=F.relu(self.fc3(result))
        result=self.fc4(result)
        return result
