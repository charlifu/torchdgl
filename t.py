import torch
import dgl
from dgl.data import CoraGraphDataset
torch.set_num_threads(1)
torch.classes.load_library("build/libTorchDGLGraph.so")
print(torch.classes.loaded_libraries)

g = CoraGraphDataset()[0]
src, dst = g.edges()

s = torch.classes.my_classes.TorchDGLGraph(src, dst)
TorchDGLGraph = torch.classes.my_classes.TorchDGLGraph


def do_in_degrees(s: TorchDGLGraph, vids: torch.Tensor):
    return s.in_degrees(vids);

scripted_func = torch.jit.script(do_in_degrees)
print(scripted_func.graph)
print("##########DGL#########")
print(do_in_degrees(g, torch.tensor([1,3,5,7])))
print("##########TorchDGL#########")
print(do_in_degrees(s, torch.tensor([1,3,5,7])))
print("##########TorchScript DGL#########")
print(scripted_func(s, torch.tensor([1,3,5,7])))

scripted_func.save("in_degree.pt")