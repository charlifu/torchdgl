import torch as th
import dgl
import dgl.function as fn
from dgl.data import CoraGraphDataset
from typing import Optional
import time
th.set_num_threads(1)
th.classes.load_library("build/libadjmatrix.so")
# torch.ops.load_library("build/libadjmatrix.so")

g = CoraGraphDataset()[0]
src, dst = g.edges()

ufeat1 = th.ones((g.num_src_nodes(), 32))
ufeat1.requires_grad_()
ufeat2 = th.ones((g.num_src_nodes(), 32))
ufeat2.requires_grad_()
g.srcdata["h"] = ufeat1
run_time = 0.0
for _ in range(10):
    start_run = time.perf_counter()
    g.update_all(fn.copy_src(src="h", out="m"), 
        fn.sum(msg="m", out="h"))
    run_time += (time.perf_counter() - start_run)

print('DGL Time (ms): {:.3f}'.format(run_time*1e3/10))
s = th.classes.DGL.AdjMatrix(src, dst)
AdjMatrix = th.classes.DGL.AdjMatrix

def do_spmm(adj: AdjMatrix, 
            op: str, 
            reduce: str, 
            ufeat : Optional[th.Tensor],
            efeat : Optional[th.Tensor]):
    return th.ops.DGL.GSpMM(adj, op, reduce, ufeat, efeat)

scripted_func = th.jit.script(do_spmm)
scripted_func(s, "copy_lhs", "sum", ufeat2, None)
run_time = 0.0
for _ in range(10):
    start_run = time.perf_counter()
    scripted_func(s, "copy_lhs", "sum", ufeat2, None)
    run_time += (time.perf_counter() - start_run)

print('Script Time (ms): {:.3f}'.format(run_time*1e3/10))
# print(scripted_func.graph)
# print("##########DGL#########")
# rst1 = th.sum(g.dstdata["h"])
# rst1.backward()
# print(ufeat1.grad)
# print("##########AdjMatrix#########")
# tmp = do_spmm(s, "copy_lhs", "sum", ufeat2, None)
# rst2 = th.sum(tmp)
# rst2.backward()
# print(ufeat2.grad)
# print("##########TorchScript DGL#########")
#print(scripted_func(s, "copy_lhs", "sum", ufeat, None))

# scripted_func.save("in_degree.pt")
