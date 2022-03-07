import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GATConv


class GAT_DGL(nn.Module):
    def __init__(self, 
                 g,
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 dropout):
        super(GAT_DGL, self).__init__()
        self.g = g
        self.layer1 = GATConv(in_dim, hidden_dim,
                              feat_drop=dropout,
                              attn_drop=dropout,
                              num_heads=1, bias=False)
        self.layer2 = GATConv(hidden_dim, out_dim,
                              feat_drop=dropout,
                              attn_drop=dropout,
                              num_heads=1, bias=False)

    def forward(self, features):
        h = self.layer1(self.g, features)
        h = F.elu(h)
        h = self.layer2(self.g, h)
        return h