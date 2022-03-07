import argparse, time, math
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
from torch_sparse import SparseTensor

import GAT_DGL, GAT_PyG, GAT_Script

def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features).view(features.shape[0], -1)
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

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

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

    # normalization
    model = None

    src, dst = g.edges()
    if args.impl == "script":
        adj = GAT_Script.AdjMatrix(src, dst)
        m = GAT_Script.GAT_Adj(adj,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.dropout)
        if cuda:
            m.cuda()
        model = th.jit.script(m)
        print(model.graph)

    elif args.impl == "dgl":
        model = GAT_DGL.GAT_DGL(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.dropout)
        if cuda:
            model.cuda()

    elif args.impl == "pyg":
        adj_pyg = SparseTensor(row=src, col=dst, sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))

        model = GAT_PyG.GAT_PyG(adj_pyg,
                        in_feats,
                        args.n_hidden,
                        n_classes,
                        args.dropout)
        if cuda:
            model.cuda()

    loss_fcn = th.nn.CrossEntropyLoss()

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
        # if epoch == 100:
        #     th.cuda.nvtx.range_push("forward")
        logits = model(features).view(features.shape[0], -1)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        # if epoch == 100:
        #     th.cuda.nvtx.range_pop()
        optimizer.zero_grad()
        # if epoch == 100:
        #     th.cuda.nvtx.range_push("backward")
        loss.backward()
        optimizer.step()

        if args.gpu > 0:
            th.cuda.synchronize()
        # if epoch == 100:
        #     th.cuda.nvtx.range_pop()

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
    parser.add_argument("--impl", type=str, default="dgl",
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