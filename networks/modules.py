import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, z,text_features,t):
        # text_features[bs,128]
        # z = [bs，128]
        x = torch.cat([z, text_features], dim=-1)
        # x = [bs,256]
        j = 0
        for i in x:
            T = t[j]
            processed = self.mlp(i).unsqueeze(0)#[1,]

        
        pass
       