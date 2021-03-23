#ifndef INCLUDE_TORCH_DGL_GARPH_H_
#define INCLUDE_TORCH_DGL_GARPH_H_

#include <dgl/aten/coo.h>
#include <dgl/aten/types.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>
#include <heterograph.h>

#include <ATen/core/TensorBody.h>
#include <torch/script.h>
#include <torch/custom_class.h>

#include <ATen/DLConvertor.h>
struct TorchDGLGraph : torch::CustomClassHolder {
  dgl::HeteroGraphPtr hgptr;
  //   dgl::HeteroGraph hg_;
  //   TorchDGLGraph(dgl::HeteroGraph hg) : hg_(hg) {}
public:
  TorchDGLGraph(at::Tensor src, at::Tensor dst);
  TorchDGLGraph(dgl::HeteroGraphPtr hgptr): hgptr(hgptr){};
  torch::Tensor in_degrees(torch::Tensor vids);
};

#endif /* INCLUDE_TORCH_DGL_GARPH_H_ */
