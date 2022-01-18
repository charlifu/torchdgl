#ifndef INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_
#define INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_


#ifndef DGL_USE_CUDA
#define DGL_USE_CUDA
#endif

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

    bool reversed_;

    AdjMatrix(torch::Tensor src, torch::Tensor dst);

    AdjMatrix(std::shared_ptr<dgl::aten::CSRMatrix> csc,
            std::shared_ptr<dgl::aten::CSRMatrix> csr,
            std::shared_ptr<dgl::aten::COOMatrix> coo, 
            dgl::dgl_format_code_t format) : csc_(csc), csr_(csr), coo_(coo), 
            format_(format), current_format_(0), 
            csc_defined_(false), csr_defined_(false), coo_defined_(false) {
        if (csc != nullptr) {
            current_format_ |= dgl::CSC_CODE;
            csc_defined_ = true;
        }
        if (csr != nullptr) {
            current_format_ |= dgl::CSR_CODE;
            csr_defined_ = true;
        }
        if (coo != nullptr) {
            current_format_ |= dgl::COO_CODE;
            coo_defined_ = true;
        }
    }

    inline DLContext Context() {
        if (csc_defined_)
            return csc_->indptr->ctx;
        else if (csr_defined_)
            return csr_->indptr->ctx;
        return coo_->row->ctx;
    }

    inline DLDataType DataType() {
        if (csc_defined_)
            return csc_->indptr->dtype;
        else if (csr_defined_)
            return csr_->indptr->dtype;
        return coo_->row->dtype;
    }

    inline int64_t src_num() {
        if (csc_defined_)
            return csc_->num_cols;
        else if (csr_defined_)
            return csr_->num_rows;
        return coo_->num_rows;
    }

    inline int64_t dst_num() {
        if (csc_defined_)
            return csc_->num_rows;
        else if (csr_defined_)
            return csr_->num_cols;
        return coo_->num_cols;
    }

    // AdjMatrix reverse() {
    //     auto newcoo = (this->coo_defined_ ? 
    //         std::make_shared<dgl::aten::COOMatrix>(dgl::aten::COOTranspose(*this->coo_)) : nullptr);
    //     dgl::dgl_format_code_t newformat = (dgl::dgl_format_code_t)0U;
    //     if (this->format_ & dgl::CSC_CODE)
    //         newformat |= dgl::CSR_CODE;
    //     if (this->format_ & dgl::CSR_CODE)
    //         newformat |= dgl::CSC_CODE;
    //     if (this->format_ & dgl::COO_CODE)
    //         newformat |= dgl::COO_CODE;
    //     return AdjMatrix(this->csr_, this->csc_, newcoo, newformat);
    // }

    void toggle_reversed() {
        this->reversed_ = !(this->reversed_);
    }

    dgl::SparseFormat SelectFormat(dgl::dgl_format_code_t preferred_format);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSCMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSRMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::COOMatrix> GetCOOMatrix(bool inplace = true);
    inline std::shared_ptr<dgl::aten::CSRMatrix> GetCSC(bool inplace = true);
    inline std::shared_ptr<dgl::aten::CSRMatrix> GetCSR(bool inplace = true);
    inline std::shared_ptr<dgl::aten::COOMatrix> GetCOO(bool inplace = true);
    torch::Tensor indegrees(c10::optional<torch::Tensor> node_ids = c10::nullopt);
    torch::Tensor outdegrees(c10::optional<torch::Tensor> node_ids = c10::nullopt);

    void SpMM(const std::string & op, const std::string & reduce,
                        dgl::NDArray ufeat, dgl::NDArray efeat,
                        dgl::NDArray out, std::vector<dgl::NDArray> out_aux);
    
    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
                _gspmm(const std::string & op, const std::string & reduce,
                        c10::optional<torch::Tensor> ufeat, c10::optional<torch::Tensor> efeat);
    torch::Tensor gspmm(std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat, c10::optional<torch::Tensor> efeat);
};

#endif // INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_
