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
    e = th.randn(size=(g.number_of_edges(), 16))
    coff = th.randn(size=(g.number_of_edges(), 16))
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        e = e.cuda(args.gpu)
        coff = coff.cuda(args.gpu)
        g = g.to(args.gpu)

    # features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    n_classes = data.num_classes

    # add self loop
    n_edges = g.number_of_edges()
    src, dst = g.edges()
    adj = AdjMatrix(src, dst)

    e1 = e.clone().detach()
    e2 = e.clone().detach()
    coff1 = coff.clone().detach()
    coff2 = coff.clone().detach()
    e1.requires_grad_()
    e2.requires_grad_()

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

    print("Forward:")

    rst1 = dgl.ops.edge_softmax(g, e1)
    rst2 = th.ops.DGL.EdgeSoftmax(adj, e2, "dst")

    if not th.allclose(rst1, rst2):
        print("Wrong results")
        sys.exit(-1)
    
    print("Backward: ")

    tmp1 = rst1 * coff1 
    tmp2 = rst2 * coff2

    sum1 = th.sum(tmp1)
    sum2 = th.sum(tmp2)

    sum1.backward()
    sum2.backward()

    cmp1 = th.allclose(e1.grad, e2.grad)

    if not cmp1:
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