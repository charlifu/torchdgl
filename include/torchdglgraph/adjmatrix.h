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

    inline int64_t edge_num() {
        if (csc_defined_)
            return csc_->indices->shape[0];
        else if (csr_defined_)
            return csr_->indices->shape[0];
        else
            return coo_->row->shape[0];
    }

    void toggle_reversed() {
        this->reversed_ = !(this->reversed_);
    }

    dgl::SparseFormat SelectFormat(dgl::dgl_format_code_t preferred_format);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSCMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSRMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::COOMatrix> GetCOOMatrix(bool inplace = true);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSC(bool inplace = true);
    std::shared_ptr<dgl::aten::CSRMatrix> GetCSR(bool inplace = true);
    std::shared_ptr<dgl::aten::COOMatrix> GetCOO(bool inplace = true);
    torch::Tensor indegrees(c10::optional<torch::Tensor> node_ids = c10::nullopt);
    torch::Tensor outdegrees(c10::optional<torch::Tensor> node_ids = c10::nullopt);

    void SpMM(const std::string & op, const std::string & reduce,
                        dgl::NDArray ufeat, dgl::NDArray efeat,
                        dgl::NDArray out, std::vector<dgl::NDArray> out_aux);

    void SDDMM(const std::string & op, dgl::NDArray lhs, dgl::NDArray rhs, 
                dgl::NDArray out, int lhs_target, int rhs_target);

    void Edge_softmax_forward(const std::string & op,
                            dgl::NDArray ufeat,
                            dgl::NDArray efeat,
                            dgl::NDArray out);
    void Edge_softmax_backward(const std::string& op,
                            NDArray out,
                            NDArray sds,
                            NDArray back_out,
                            NDArray ufeat);
    
    torch::Tensor _edge_softmax_forward(c10::optional<torch::Tensor> e,
                                        const std::string & op);
    torch::Tensor _edge_softmax_backward(c10::optional<torch::Tensor> out,
                                        c10::optional<torch::Tensor> sds);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
                _gspmm(const std::string & op, const std::string & reduce,
                        c10::optional<torch::Tensor> ufeat, c10::optional<torch::Tensor> efeat);
    torch::Tensor gspmm(std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat, c10::optional<torch::Tensor> efeat);

    torch::Tensor _gsddmm(const std::string & op, c10::optional<torch::Tensor> lhs,
                            c10::optional<torch::Tensor> rhs,
                            std::string lhs_target = "u", std::string rhs_target = "v");
    torch::Tensor gsddmm(std::string op, c10::optional<torch::Tensor> lhs_data,
                            c10::optional<torch::Tensor> rhs_data,
                            std::string lhs_target = "u", std::string rhs_target = "v");
};

#endif // INCLUDE_TORCHDGLGRAPH_ADJMATRIX_H_
