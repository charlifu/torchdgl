#include <dgl/aten/array_ops.h>
#include <torchdglgraph/adjmatrix.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/custom_class.h>

#include <ATen/core/function_schema.h>

#include <ATen/core/op_registration/infer_schema.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <string>
#include <vector>

dgl::NDArray ToNDArray(at::Tensor t) {
  return dgl::NDArray::FromDLPack(at::toDLPack(t));
}

AdjMatrix::AdjMatrix(torch::Tensor src, torch::Tensor dst) {
    int64_t num_src = at::max(src).item().toLong() + 1;
    int64_t num_dst = at::max(dst).item().toLong() + 1;

    format_ = dgl::CSR_CODE | dgl::CSC_CODE | dgl::COO_CODE;
    current_format_ = dgl::COO_CODE;

    coo_defined_ = true;
    csr_defined_ = false;
    csc_defined_ = false;

    coo_ = std::make_shared<dgl::aten::COOMatrix>(
            num_src, num_dst, ToNDArray(src), ToNDArray(dst));
    csr_ = nullptr;
    csc_ = nullptr;
}

std::shared_ptr<dgl::aten::CSRMatrix> AdjMatrix::GetCSCMatrix(bool inplace) {
    if (inplace && !(format_ & dgl::CSC_CODE))
        return nullptr;
    // Prefers converting from COO since it is parallelized.
    if (!csc_defined_) {
        if (coo_defined_) {
            const auto & newadj = dgl::aten::COOToCSR(
                    dgl::aten::COOTranspose(*coo_));

            if (inplace) {
                this->csc_ = std::make_shared<dgl::aten::CSRMatrix>(newadj);
                csc_defined_ = true;
                current_format_ |= dgl::CSC_CODE;
            }
            else
                return std::make_shared<dgl::aten::CSRMatrix>(newadj);
        } else {
            if (!csr_defined_)
                return nullptr;
            const auto & newadj = dgl::aten::CSRTranspose(*csr_);

            if (inplace) {
                this->csc_ = std::make_shared<dgl::aten::CSRMatrix>(newadj);
                csc_defined_ = true;
                current_format_ |= dgl::CSC_CODE;
            }
            else
                return std::make_shared<dgl::aten::CSRMatrix>(newadj);
        }
    }
    return this->csc_;
}

std::shared_ptr<dgl::aten::CSRMatrix> AdjMatrix::GetCSRMatrix(bool inplace) {
    if (inplace && !(format_ & dgl::CSR_CODE))
        return nullptr;
    // Prefers converting from COO since it is parallelized.
    if (!csr_defined_) {
        if (coo_defined_) {
            const auto & newadj = dgl::aten::COOToCSR(*coo_);

            if (inplace) {
                this->csr_ = std::make_shared<dgl::aten::CSRMatrix>(newadj);
                csr_defined_ = true;
                current_format_ |= dgl::CSR_CODE;
            }
            else
                return std::make_shared<dgl::aten::CSRMatrix>(newadj);
        } else {
            if (!csc_defined_)
                return nullptr;
            const auto & newadj = dgl::aten::CSRTranspose(*csc_);

            if (inplace) {
                this->csr_ = std::make_shared<dgl::aten::CSRMatrix>(newadj);
                csr_defined_ = true;
                current_format_ |= dgl::CSR_CODE;
            }
            else
                return std::make_shared<dgl::aten::CSRMatrix>(newadj);
        }
    }
    return this->csr_;
}

std::shared_ptr<dgl::aten::COOMatrix> AdjMatrix::GetCOOMatrix(bool inplace) {
    if (inplace && !(format_ & dgl::COO_CODE))
        return nullptr;
    // Prefers converting from COO since it is parallelized.
    if (!coo_defined_) {
        if (csc_defined_) {
            const auto & newadj = dgl::aten::COOTranspose(dgl::aten::CSRToCOO(*csc_, true));

            if (inplace) {
                this->coo_ = std::make_shared<dgl::aten::COOMatrix>(newadj);
                coo_defined_ = true;
                current_format_ |= dgl::COO_CODE;
            }
            else
                return std::make_shared<dgl::aten::COOMatrix>(newadj);
        } else {
            if (!csr_defined_)
                return nullptr;
            const auto & newadj = dgl::aten::CSRToCOO(*csr_, true);

            if (inplace) {
                this->coo_ = std::make_shared<dgl::aten::COOMatrix>(newadj);
                coo_defined_ = true;
                current_format_ |= dgl::COO_CODE;
            }
            else
                return std::make_shared<dgl::aten::COOMatrix>(newadj);
        }
    }
    return this->coo_;
}

dgl::SparseFormat AdjMatrix::SelectFormat(dgl::dgl_format_code_t preferred_format) {
    dgl::dgl_format_code_t common = preferred_format & format_;
    if (common & current_format_)
        return dgl::DecodeFormat(common & current_format_);

    if (common)
        return dgl::DecodeFormat(common);
    return dgl::DecodeFormat(current_format_);
}

torch::Tensor indegree(const c10::intrusive_ptr<AdjMatrix> & adj, torch::Tensor node_ids) {
    dgl::SparseFormat fmt = adj->SelectFormat(dgl::CSC_CODE);
    auto vids = ToNDArray(node_ids);
    dgl::NDArray degs;
    if (fmt == dgl::SparseFormat::kCSC) {
        degs = dgl::aten::CSRGetRowNNZ(*adj->GetCSCMatrix(), vids);
    }
    else if (fmt == dgl::SparseFormat::kCOO)
        degs = dgl::aten::COOGetRowNNZ(dgl::aten::COOTranspose(*adj->GetCOOMatrix()), vids);
    return at::fromDLPack(degs.ToDLPack());
}

torch::Tensor outdegree(const c10::intrusive_ptr<AdjMatrix> & adj, torch::Tensor node_ids) {
    dgl::SparseFormat fmt = adj->SelectFormat(dgl::CSR_CODE);
    auto vids = ToNDArray(node_ids);
    dgl::NDArray degs;
    if (fmt == dgl::SparseFormat::kCSR)
        degs = dgl::aten::CSRGetRowNNZ(*adj->GetCSRMatrix(), vids);
    else if (fmt == dgl::SparseFormat::kCOO)
        degs = dgl::aten::COOGetRowNNZ(*adj->GetCOOMatrix(), vids);
    return at::fromDLPack(degs.ToDLPack());
}

TORCH_LIBRARY(my_classes, m) {
  m.class_<AdjMatrix>("AdjMatrix")
      .def(torch::init<torch::Tensor, torch::Tensor>());
  m.def("indegree", indegree);
  m.def("outdegree", outdegree);
}

