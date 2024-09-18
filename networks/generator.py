import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import graph_kit
# import graph_kit.graph_kit
from . import tgcn 
# import ConvTemporalGraphical
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 一个简单模块，为生成的图像注入噪声。
# 包含一个学习参数，该参数会与噪声相乘后添加到图像上。
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

# `Mapping_Net`：一个多层感知器（MLP），用于处理潜在空间向量。
class Mapping_Net(nn.Module):
    def __init__(self, latent=1024, mlp=4):
        super().__init__()

        layers = []
        for i in range(mlp):
            linear = nn.Linear(latent, latent)
            linear.weight.data.normal_()
            linear.bias.data.zero_()
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_classes, t_size, mlp_dim=4, edge_importance_weighting=True, dataset='kit', device = 'cuda',**kwargs):
        super().__init__()

        # load graph
        if dataset == 'kit':
            self.graph =graph_kit.graph_kit() 
            # print(self.graph,type(self.graph))  
            self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False) for Al in self.graph.As]
        # print(len(self.A),type(self.A))
        # self.A.to(device)
        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for i, _ in enumerate(self.A)]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        self.mlp = Mapping_Net(in_channels+n_classes, mlp_dim)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 512, kernel_size, 1, graph=self.graph, lvl=3, bn=False, residual=False, up_s=False, up_t=1, **kwargs),
            st_gcn(512, 256, kernel_size, 1, graph=self.graph, lvl=3, up_s=False, up_t=int(self.t_size/16), **kwargs),
            st_gcn(256, 128, kernel_size, 1, graph=self.graph, lvl=2, bn=False, up_s=True, up_t=int(self.t_size/16), **kwargs),
            st_gcn(128, 64, kernel_size, 1, graph=self.graph, lvl=2, up_s=False, up_t=int(self.t_size/8), **kwargs),
            st_gcn(64, 32, kernel_size, 1, graph=self.graph, lvl=1, bn=False, up_s=True, up_t=int(self.t_size/4), **kwargs),
            st_gcn(32, out_channels, kernel_size, 1, graph=self.graph, lvl=1, up_s=False, up_t=int(self.t_size/2), **kwargs),
            st_gcn(out_channels, out_channels, kernel_size, 1, graph=self.graph, lvl=0, bn=False, up_s=True, up_t=self.t_size, tan=True, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # self.label_emb = nn.Embedding(n_classes, n_classes)
        

    def forward(self, x, labels, t_size,trunc=None):
        # labels = labels.mean(dim=1)
        c = labels.mean(dim=1)
        print("c.size()",c.size())#[batch_size,3072]
        x = torch.cat((c, x), -1)
        # print("x",x.size())#([4, word_embs+latent_dim])
        w = []
        for i in x:
            w = self.mlp(i).unsqueeze(0) if len(w)==0 else torch.cat((w, self.mlp(i).unsqueeze(0)), dim=0)
           # 这段代码的作用是对 w 向量列表应用 Truncation Trick，通过将每个向量 w[i] 向 W 空间中的均值向量 m 靠拢，从而控制生成样本的多样性与质量之间的权衡。通过调节 truncation 参数，可以生成不同质量和多样性的样本。
        w = self.truncate(w, 1000, trunc) if trunc is not None else w  # Truncation trick on W
        x = w.view((*w.shape, 1, 1))
        # print("x",x.size())
        i = 0
        out = []
        for data in x:
            # print(type(data),data.unsqueeze(0).size())
            data = data.unsqueeze(0)
            t = t_size[i]
            up_t = torch.tensor([1,int(t/16),int(t/16),int(t/8),int(t/4),int(t/2),t]).to('cuda')

            # print(up_t)
            j = 0
            # up_t.to('cuda:0')
            # data.to('cuda:0')
            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                if up_t[j] == 0:
                    up_t[j] = -1
                A = self.A[gcn.lvl].to('cuda:0')
                data,_ = gcn(data, A * importance,up_t[j])
                # print("j=",j," == ",data.size())
                j = j+1
            # print("data",data.size())
            out.append(data.squeeze(0).permute(1,2,0))
            i = i+1
        return out

    def truncate(self, w, mean, truncation):  # Truncation trick on W
        t = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (mean, *w.shape[1:]))))
        w_m = []
        for i in t:
            w_m = self.mlp(i).unsqueeze(0) if len(w_m)==0 else torch.cat((w_m, self.mlp(i).unsqueeze(0)), dim=0)

        m = w_m.mean(0, keepdim=True)

        for i,_ in enumerate(w):
            w[i] = m + truncation*(w[i] - m)

        return w

class st_gcn(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                graph=None,
                lvl=3,
                dropout=0,
                bn=True,
                residual=True,
                up_s=False, 
                up_t=64, 
                tan=False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.up_s, self.up_t, self.tan = graph, lvl, up_s, up_t, tan
        self.gcn = tgcn.ConvTemporalGraphical(in_channels, out_channels,kernel_size[1][lvl])

        tcn = [nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            )]
        
        tcn.append(nn.BatchNorm2d(out_channels)) if bn else None

        self.tcn = nn.Sequential(*tcn)


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.noise = NoiseInjection(out_channels)

        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh   = nn.Tanh()

    def forward(self, x, A,t_size = -1):

        x = self.upsample_s(x) if self.up_s else x
        # print("forward",x.size())
        if t_size == -1:
            x = F.interpolate(x, size=(self.up_t,x.size(-1)))  # Exactly like nn.Upsample
        else:
            x = F.interpolate(x, size=(t_size,x.size(-1)))  # Exactly like nn.Upsample

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res
        
        # Noise Inject
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device='cuda:0')
        x     = self.noise(x, noise)

        return self.tanh(x) if self.tan else self.l_relu(x), A

    
    def upsample_s(self, tensor):

        ids  = []
        mean = []
        for umap in self.graph.mapping[self.lvl]:
            ids.append(umap[0])
            tmp = None
            for nmap in umap[1:]:
                tmp = torch.unsqueeze(tensor[:, :, :, nmap], -1) if tmp == None else torch.cat([tmp, torch.unsqueeze(tensor[:, :, :, nmap], -1)], -1)

            mean.append(torch.unsqueeze(torch.mean(tmp, -1) / (2 if self.lvl==2 else 1), -1))

        for i, idx in enumerate(ids): tensor = torch.cat([tensor[:,:,:,:idx], mean[i], tensor[:,:,:,idx:]], -1)


        return tensor
    

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MotionLenEstimator_2(nn.Module):
    def __init__(self, word_size, hidden_size, output_size):
        super(MotionLenEstimator_2, self).__init__()

        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, cap_lens):
        num_samples = word_embs.shape[0]
        inputs = word_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        # print("cap_lens",type(cap_lens),cap_lens)#[5, 11, 11, 16]
        # pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
        # input: 填充后的序列张量。形状通常是 (seq_len, batch, features) 或 (batch, seq_len, features)，具体取决于 batch_first 的设置。
        # lengths: 一个包含每个序列实际长度的张量或列表。它的形状通常是 (batch,)。
        # batch_first: 如果为 True，input 的形状应为 (batch, seq_len, features)；如果为 False，input 的形状应为 (seq_len, batch, features)。
        # enforce_sorted: 如果为 True，输入序列必须按降序排序（即较长的序列在前）。如果为 False，排序不是必须的，但会增加一些额外的计算开销。
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True,enforce_sorted=False)
        # print("emb",emb.data.size())#([4, 16, 512]
        # print(hidden.size())#[2, 4, 512])
        gru_seq, gru_last = self.gru(emb,hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        # print("gru_last",gru_last,gru_last.size)#torch.Size([4, 1024])
        return self.output(gru_last)

# if __name__=='__main__':
#     cuda = torch.cuda.is_available()
#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#     batch_size = 4
#     latent_dim = 512
#     model = Generator(latent_dim, out_channels=3, n_classes=3072, t_size=64)
#     model.to('cuda:0')
#     labels = torch.tensor([0,1,0,0])
#     labels = Variable(labels.type(LongTensor))
#     z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
#     t_size = [28,60,80,100]
#     print(z.size())#[4,512]
#     print(labels.size())#[4]
#     fake_imgs = model(z, labels,t_size)
#     print(type(fake_imgs))#[4, 3, 64, 21]