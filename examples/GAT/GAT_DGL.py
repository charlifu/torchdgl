from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch as th

import dgl.function as fn
from dgl.nn.functional import edge_softmax

from dgl.nn.pytorch.conv import GATConv

class GATConv_DGL(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                dropout):
        super(GATConv_DGL, self).__init__() 
        self.fc_weight = nn.Parameter(th.FloatTensor(in_dim, out_dim))
        self.attn_l = nn.Parameter(th.FloatTensor(1, out_dim))
        self.attn_r = nn.Parameter(th.FloatTensor(1, out_dim))
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
    
    def forward(self,
                graph,
                features):
        h = self.feat_drop(features)
        h = th.mm(h, self.fc_weight)
        el = (h * self.attn_l).sum(-1).unsqueeze(-1)
        er = (h * self.attn_r).sum(-1).unsqueeze(-1)
        graph.srcdata.update({'ft': h, 'el': el})
        graph.dstdata.update({'er': er})
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                            fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        return rst

class GAT_DGL(nn.Module):
    def __init__(self, 
                 g,
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 dropout):
        super(GAT_DGL, self).__init__()
        self.g = g
        self.layer1 = GATConv_DGL(in_dim, hidden_dim,
                              dropout)
        self.layer2 = GATConv_DGL(hidden_dim, out_dim,
                              dropout)

    def forward(self, features):
        h = self.layer1(self.g, features)
        h = F.elu(h)
        h = self.layer2(self.g, h)
        return h