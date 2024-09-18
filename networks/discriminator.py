import torch.nn as nn
import torch
import numpy as np
from networks.graph_kit import graph_kit
from networks.STGCN import STConvBlock
from utils.utility import calc_gso
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels,Kt,Ks, num_classes = 2,device = 'cuda:0',edge_importance_weighting=True,gso_type='sym_norm_adj',dataset='kit', **kwargs):
        super().__init__()
        # load graph
        if dataset == 'kit':
            kit_adj = np.array([
                [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
            ])
        # print("xxx",self.A.shape)
        gso = calc_gso(kit_adj, gso_type)
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        gso = torch.tensor(gso,dtype=torch.float32)
        print("gso.size",type(gso))
        self.device = device

        self.enc = nn.ModuleList((
            STConvBlock(Kt, Ks, n_vertex=gso.shape[0],in_channels=in_channels,out_channels= 32,act_func='glu',graph_conv_type='graph_conv',gso=gso,bias=False,droprate = 0.3,residual = False),
            STConvBlock(Kt, Ks, n_vertex=gso.shape[0],in_channels=32,out_channels= 64,act_func='glu',graph_conv_type='graph_conv',gso=gso,bias=False,droprate = 0.3,residual = True),
            STConvBlock(Kt, Ks, n_vertex=gso.shape[0],in_channels=64,out_channels= out_channels,act_func='glu',graph_conv_type='graph_conv',gso=gso,bias=False,droprate = 0.3,residual = True)
        ))
        # self.mean = torch.mean()
        self.fc1 = nn.Linear(8400, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # 输出 1 个值
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数用于输出概率值

    def forward(self,x,labels):
        '''
        params:
            x:[bs,time_step,c_in*n_vertex]
            labels:[bs,dim_word]
        '''
        # print("discrminator x",type(x))
        ans = []
        for i,data in enumerate(x):
            # print("disc data",data.shape)#[74, 21, 3]
            data = torch.tensor(data).unsqueeze(0)
            # print("disc data",data.size())
            B,T,V,C = data.size()
            data.to(self.device)
            data = data.permute(0,3,1,2)
            for layer in self.enc:
                data = layer(data)
                # print(x.size())
            data = torch.nn.functional.adaptive_avg_pool2d(data, (200, 21))  # (batch_size, out_channels, target_time_steps, J)
            data = data.reshape(1,-1)
            print("after",data.size())
            data = self.fc1(data)
            data = self.relu(data)
            data = self.fc2(data)
            data = self.sigmoid(data)
            ans.append(data)
        # print("after forward:",x.size())[bs,out_channels,Time_step,J]

        return torch.tensor(ans)