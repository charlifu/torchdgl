"""GCN using basic message passing

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import argparse, time, math
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset  
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
th.classes.load_library("build/libadjmatrix.so")
AdjMatrix = th.classes.DGL.AdjMatrix

class GCNConv_pyg(MessagePassing):
    def __init__(self, 
                 norm,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=False):
        super(GCNConv_pyg, self).__init__(aggr="add")
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.norm = norm
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        if self.dropout:
            x = self.dropout(x)
        out = th.mm(x, self.weight)
        out = out * self.norm
        out = self.propagate(edge_index, x=out)
        out = out * self.norm
        if self.activation:
            out = self.activation(out)
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

class GCN_pyg(nn.Module):
    def __init__(self,
                 edge_index,
                 norm,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_pyg, self).__init__()
        self.edge_index = edge_index
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv_pyg(norm, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv_pyg(norm, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNConv_pyg(norm, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h, self.edge_index)
        return h

class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=False):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = th.mm(h, self.weight)
        self.g.ndata["h"] = h * self.g.ndata["norm"]
        self.g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        h = self.g.ndata.pop('h')
        h = h * self.g.ndata["norm"]
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h

class GCN_Adj(nn.Module):

    adj: AdjMatrix
    norm: th.Tensor

    def __init__(self,
                adj,
                norm,
                in_feats,
                n_hidden,
                n_classes,
                activation,
                dropout):
        super(GCN_Adj, self).__init__()
        self.adj = adj
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
        h = th.ops.DGL.GSpMM(self.adj, "copy_lhs", "sum", h, None)
        h = h * self.norm
        h = self.activation(h)

        # Layer 2
        h = self.dropout(h)
        h = th.mm(h, self.weight2)
        h = h * self.norm
        h = th.ops.DGL.GSpMM(self.adj, "copy_lhs", "sum", h, None)
        h = h * self.norm

        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    src, dst = g.edges()
    adj = AdjMatrix(src, dst)
    m = GCN_Adj(adj,
                    g.ndata['norm'],
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    F.relu,
                    args.dropout)
    if cuda:
        m.cuda()
    model1 = th.jit.script(m)

    model2 = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    if cuda:
        model2.cuda()

    adj_pyg = SparseTensor(row=src, col=dst, sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))

    model3 = GCN_pyg(adj_pyg,
                    g.ndata['norm'],
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)
    if cuda:
        model3.cuda()

    loss_fcn = th.nn.CrossEntropyLoss()
    model = None

    # create GCN model
    if args.compile == "script":
        model = model1
        print("Using Torchscript")
        print(model.graph)

    elif args.compile == "pyg":
        model = model3
        print("Using Pyg")

    else:
        model = model2
        print("Using DGL")

    optimizer = th.optim.Adam(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur) * 1000, loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--compile", type=str, default="dgl",
            help="use torch script or not")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)