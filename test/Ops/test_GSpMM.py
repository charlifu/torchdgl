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
import sys

th.classes.load_library("../../build/libadjmatrix.so")
th.ops.load_library("../../build/libadjmatrix.so")
AdjMatrix = th.classes.DGL.AdjMatrix


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
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    features = th.randn(size=(g.number_of_nodes(), 16))
    efeat = th.randn(size=(g.number_of_edges(), 1))
    coff = th.rand_like(features)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        features = features.cuda(args.gpu)
        efeat = efeat.cuda(args.gpu)
        coff = coff.cuda(args.gpu)
        g = g.to(args.gpu)

    # features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes

    # add self loop
    n_edges = g.number_of_edges()
    n_vertices = g.number_of_nodes()
    src, dst = g.edges()
    adj = AdjMatrix(src, dst)

    features.requires_grad_()
    efeat.requires_grad_()

    feat1 = features.clone().detach()
    feat2 = features.clone().detach()
    efeat1 = efeat.clone().detach()
    efeat2 = efeat.clone().detach()
    coff1 = coff.clone().detach()
    coff2 = coff.clone().detach()
    feat1.requires_grad_()
    efeat1.requires_grad_()
    feat2.requires_grad_()
    efeat2.requires_grad_()

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
    
    ops = [
        "copy_lhs",
        "copy_rhs",
        "add",
        "mul"
    ]

    reduce_ops = [
        "sum",
        "max",
        "min"
    ]

    for op in ops:
        for reduce in reduce_ops:

            print(op, reduce)

            rst1 = dgl.ops.gspmm(g, op, reduce, feat1, efeat1)
            rst2 = th.ops.DGL.GSpMM(adj, op, reduce, feat2, efeat2)

            if not th.allclose(rst1, rst2):
                print(f"{op}, {reduce}: Wrong results")
                sys.exit(-1)
            
            rst1 = rst1 * coff1
            rst2 = rst2 * coff2

            rst1 = th.sum(rst1)
            rst2 = th.sum(rst2)

            rst1.backward()
            rst2.backward()

            if not th.allclose(feat1.grad, feat2.grad):
                print("Wrong backward results")
                sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    args = parser.parse_args()
    print(args)

    main(args)