#ifndef INCLUDE_TORCHDGLGRAPH_TORCH_DGL_GRAPH_H_
#define INCLUDE_TORCHDGLGRAPH_TORCH_DGL_GRAPH_H_

#include <dgl/aten/coo.h>
#include <dgl/aten/types.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/base_heterograph.h>
#include <dgl/kernel.h>

#include <ATen/core/TensorBody.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <ATen/DLConvertor.h>

struct TorchDGLMetaGraph : torch::CustomClassHolder {
  dgl::GraphPtr metagraph;
  //   dgl::HeteroGraph hg_;
  //   TorchDGLGraph(dgl::HeteroGraph hg) : hg_(hg) {}
public:
  TorchDGLMetaGraph(){};
  TorchDGLMetaGraph(dgl::GraphPtr metagraph) : metagraph(metagraph){};
  // TorchDGLMetaGraph(dgl::HeteroGraphPtr hgptr): hgptr(hgptr){};
  std::array<int64_t, 2> FindEdge(int64_t etypes);
};

struct TorchDGLGraph : torch::CustomClassHolder {
  dgl::HeteroGraphPtr hgptr;
  //   dgl::HeteroGraph hg_;
  //   TorchDGLGraph(dgl::HeteroGraph hg) : hg_(hg) {}
public:
  TorchDGLGraph(){};
  TorchDGLGraph(at::Tensor src, at::Tensor dst);
  TorchDGLGraph(dgl::HeteroGraphPtr hgptr) : hgptr(hgptr){};
  torch::Tensor InDegrees(torch::Tensor vids);
  int64_t NumVertexs(int64_t etypes);
  int64_t NumEdges(int64_t etypes);
  int64_t NumETypes();
  std::string DataType();
  c10::intrusive_ptr<TorchDGLMetaGraph> GetMetaGraph();
};
#endif // INCLUDE_TORCHDGLGRAPH_TORCH_DGL_GRAPH_H_
