import argparse, time, math
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    norm: th.Tensor

    def __init__(self,
                norm,
                in_feats,
                n_hidden,
                n_classes,
                activation,
                dropout):
        super(MyModel, self).__init__()
        self.norm = norm
        self.weight1 = nn.Parameter(th.Tensor(in_feats, n_hidden)) 
        self.weight2 = nn.Parameter(th.Tensor(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, features : th.Tensor):
        # Layer 1
        h = self.dropout(features)
        h = th.mm(h, self.weight1)
        h = h * self.norm
        h = h * self.norm
        h = self.activation(h)

        # Layer 2
        h = self.dropout(h)
        h = th.mm(h, self.weight2)
        h = h * self.norm
        h = h * self.norm

        return h


def main():
    m = MyModel(3000, 16, 3, F.relu, 0.5)