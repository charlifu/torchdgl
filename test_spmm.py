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
from typing import Optional
th.classes.load_library("build/libadjmatrix.so")
AdjMatrix = th.classes.DGL.AdjMatrix

def do_spmm(adj: AdjMatrix, 
            op: str, 
            reduce: str, 
            ufeat : Optional[th.Tensor],
            efeat : Optional[th.Tensor]):
    return th.ops.DGL.GSpMM(adj, op, reduce, ufeat, efeat)

scripted_spmm = th.jit.script(do_spmm)

class GCNConv_pyg(MessagePassing):
    def __init__(self):
        super(GCNConv_pyg, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

pyg_spmm = GCNConv_pyg()

def run_dgl(g, features):
    g.ndata["h"] = features
    g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
    return g.ndata['h']

def run_pyg(edge_index, features):
    return pyg_spmm(features, edge_index)

def run_script(adj, features):
    return scripted_spmm(adj, "copy_lhs", "sum", features, None)

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
    in_feats = features.shape[1]
    n_classes = data.num_classes
    print("feature size: {}".format(in_feats))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    src, dst = g.edges()
    edge_index = SparseTensor(row=src, 
                        col=dst, 
                        sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
    adj = AdjMatrix(src, dst)

    runtime = 0.0
    n = 1

    if args.impl == "dgl":
        run_dgl(g, features)
        if args.gpu >= 0:
            th.cuda.synchronize()
        # th.cuda.nvtx.range_push("spmm start")
        for _ in range(n):
            start_run = time.perf_counter()
            run_dgl(g, features)
            if args.gpu >= 0:
                th.cuda.synchronize()
            runtime += time.perf_counter() - start_run
        # th.cuda.nvtx.range_pop()
    elif args.impl == "pyg":
        run_pyg(edge_index, features)
        if args.gpu >= 0:
            th.cuda.synchronize()
        for _ in range(n):
            start_run = time.perf_counter()
            run_pyg(edge_index, features)
            if args.gpu >= 0:
                th.cuda.synchronize()
            runtime += time.perf_counter() - start_run
    else:
        run_script(adj, features)
        if args.gpu >= 0:
            th.cuda.synchronize()
        for _ in range(n):
            start_run = time.perf_counter()
            run_script(adj, features)
            if args.gpu >= 0:
                th.cuda.synchronize()
            runtime += time.perf_counter() - start_run
    
    #print('Time (ms): {:.3f}'.format(runtime*1e3/n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--impl", type=str, default="dgl",
            help="use torch script or not")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    args = parser.parse_args()
    print(args)

    main(args)
