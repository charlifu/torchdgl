#ifndef INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_
#define INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_

#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/types.h>
#include <dgl/aten/spmat.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/kernel.h>

#include <ATen/core/TensorBody.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <ATen/DLConvertor.h>

#include <memory>

struct AdjMatrix : torch::CustomClassHolder {
    std::shared_ptr<dgl::aten::CSRMatrix> csr_, csc_;
    std::shared_ptr<dgl::aten::COOMatrix> coo_;

    dgl::dgl_format_code_t format_;
    dgl::dgl_format_code_t current_format_;
    bool csr_defined_;
    bool csc_defined_;
    bool coo_defined_;

    AdjMatrix(torch::Tensor src, torch::Tensor dst);

    dgl::SparseFormat SelectFormat(dgl::dgl_format_code_t preferred_format);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSCMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSRMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::COOMatrix> GetCOOMatrix(bool inplace = true);
};

torch::Tensor indegree(const c10::intrusive_ptr<AdjMatrix> & adj, torch::Tensor node_ids);
torch::Tensor outdegree(const c10::intrusive_ptr<AdjMatrix> & adj, torch::Tensor node_ids);

#endif // INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_
