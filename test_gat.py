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


