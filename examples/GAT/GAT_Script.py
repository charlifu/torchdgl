import torch as th
import torch.nn as nn
import torch.nn.functional as F

th.classes.load_library("../../build/libadjmatrix.so")
th.ops.load_library("../../build/libadjmatrix.so")
AdjMatrix = th.classes.DGL.AdjMatrix

class GATConv_Adj(nn.Module):

    def __init__(self,
                in_feats,
                out_feats,
                dropout):
        super(GATConv_Adj, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc_weight = nn.Parameter(th.FloatTensor(in_feats, out_feats)) 
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
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
                adj : AdjMatrix,
                features : th.Tensor):
        h = self.feat_drop(features)
        h = th.mm(h, self.fc_weight)
        el = (h * self.attn_l).sum(-1).unsqueeze(-1)
        er = (h * self.attn_r).sum(-1).unsqueeze(-1)

        e = self.leaky_relu(th.ops.DGL.GSDDMM(adj, "add", el, er, "u", "v"))
        e = self.attn_drop(th.ops.DGL.EdgeSoftmax(adj, e, "dst"))
        h = th.ops.DGL.GSpMM(adj, "mul", "sum", h, e)

        return h

class GAT_Adj(nn.Module):

    adj: AdjMatrix

    def __init__(self,
                adj,
                in_feats,
                n_hidden,
                n_classes,
                dropout):
        super(GAT_Adj, self).__init__()
        self.adj = adj
        self.layer1 = GATConv_Adj(in_feats, n_hidden,
                              dropout=dropout)
        self.layer2 = GATConv_Adj(n_hidden, n_classes,
                              dropout=dropout)

    def forward(self, features : th.Tensor):
        h = self.layer1(self.adj, features)
        h = F.elu(h)
        h = self.layer2(self.adj, h)
        return h
