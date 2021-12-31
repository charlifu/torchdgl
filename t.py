import torch
import dgl
from dgl.data import CoraGraphDataset
torch.set_num_threads(1)
torch.classes.load_library("build/libadjmatrix.so")
torch.ops.load_library("build/libadjmatrix.so")
print(torch.classes.loaded_libraries)

g = CoraGraphDataset()[0]
src, dst = g.edges()

s = torch.classes.my_classes.AdjMatrix(src, dst)
AdjMatrix = torch.classes.my_classes.AdjMatrix


def do_in_degrees(adj: AdjMatrix, vids: torch.Tensor):
    return torch.ops.my_classes.indegree(adj, vids);

scripted_func = torch.jit.script(do_in_degrees)
print(scripted_func.graph)
print("##########DGL#########")
print(g.in_degrees(torch.tensor([1,3,5,7])))
print("##########AdjMatrix#########")
print(do_in_degrees(s, torch.tensor([1,3,5,7])))
print("##########TorchScript DGL#########")
print(scripted_func(s, torch.tensor([1,3,5,7])))

scripted_func.save("in_degree.pt")
