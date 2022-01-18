#include <torchdglgraph/adjmatrix.h>
#include <dgl/aten/array_ops.h>
#include <dgl/bcast.h>
#include <array/kernel_decl.h>


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

dgl::NDArray ToNDArray(const c10::optional<at::Tensor> & t) {
    if (t) {
        return dgl::NDArray::FromDLPack(at::toDLPack(t.value()));
    } 
    else {
        return dgl::aten::NullArray();
    }
}

c10::Device get_device(DLContext ctx) {
    if (ctx.device_type == DLDeviceType::kDLCPU) 
        return c10::Device(c10::DeviceType::CPU, ctx.device_id);
    else if (ctx.device_type == DLDeviceType::kDLGPU)
        return c10::Device(c10::DeviceType::CUDA, ctx.device_id);
    return c10::Device(c10::DeviceType::CPU, ctx.device_id);
}

c10::ScalarType get_dtype(DLDataType dtype) {
    if (dtype.code == DLDataTypeCode::kDLInt) {
        if (dtype.bits == 8)
            return torch::kInt8;
        else if (dtype.bits == 32)
            return torch::kInt32;
        else if (dtype.bits == 64)
            return torch::kInt64;
        return torch::kInt;
    }
    else if (dtype.code == DLDataTypeCode::kDLFloat) {
        if (dtype.bits == 16)
            return torch::kFloat16;
        else if (dtype.bits == 32)
            return torch::kF32;
        else if (dtype.bits == 64)
            return torch::kF64;
        return torch::kFloat;
    }
    return torch::kFloat;
}


bool need_reduce_last_dim(const c10::optional<torch::Tensor> & ufeat, const c10::optional<torch::Tensor> & efeat) {
    if (!ufeat || !efeat)
        return false;
    auto ushp = ufeat.value().sizes();
    auto eshp = efeat.value().sizes();
    auto us = ushp.size(), es = eshp.size();
    return ushp.slice(1, us-2).equals(eshp.slice(1, es-2)) && eshp.back() == 1 && ushp.back() > 1;
}

bool spmm_cache_ufeat(const std::string & op, const std::string & reduce, bool req_grad_u, bool req_grad_e) {
    if (op != "copy_lhs" && req_grad_e) {
        if (reduce == "sum")
            return true;
        else {
            if (op == "mul")
                return true;
        }
    }
    return false;
}

bool spmm_cache_efeat(const std::string & op, const std::string & reduce, bool req_grad_u, bool req_grad_e) {
    if (op != "copy_rhs" && req_grad_u) {
        if (reduce == "sum") {
            if (op == "mul" || op == "add")
                return true;
        }
        else {
            if (op == "mul")
                return true;
        }
    }
    return false;
}

bool spmm_cache_argu(const std::string & op, const std::string & reduce, bool req_grad_u, bool req_grad_e) {
    if (req_grad_u || req_grad_e) {
        if (reduce == "min" || reduce == "max")
            return true;
    }
    return false;
}

bool spmm_cache_arge(const std::string & op, const std::string & reduce, bool req_grad_u, bool req_grad_e) {
    if (req_grad_u || req_grad_e) {
        if (reduce == "min" || reduce == "max")
            return true;
    }
    return false;
}

c10::IntArrayRef infer_broadcast_shape(const std::string & op, c10::IntArrayRef shp1, c10::IntArrayRef shp2) {
    auto pad_shp1 = shp1;
    auto pad_shp2 = shp2;
    if (op == "dot" && shp1.back() != shp2.back())
        LOG(FATAL) << "Dot operator is only available for arrays with the same size on last dimension, but got {} and {}.";

    if (op == "copy_lhs")
        return shp1;
    if (op == "copy_rhs")
        return shp2;
    
    if (shp1.size() > shp2.size()) {
        std::vector<int64_t> tmpvec(shp1.size() - shp2.size(), 1);
        for (int i = 0; i < shp2.size(); ++i)
            tmpvec.push_back(shp2[i]);
        pad_shp1 = c10::IntArrayRef(tmpvec);
    }
    else if (shp1.size() < shp2.size()) {
        std::vector<int64_t> tmpvec(shp2.size() - shp1.size(), 1);
        for (int i = 0; i < shp1.size(); ++i)
            tmpvec.push_back(shp1[i]);
        pad_shp1 = c10::IntArrayRef(tmpvec);
    }

    std::vector<int64_t> rst;

    for (int i = 0; i < pad_shp1.size(); ++i) {
        if (pad_shp1[i] != pad_shp2[i] && pad_shp1[i] != 1 && pad_shp2[i] != 1)
            LOG(FATAL) << "Feature shapes {} and {} are not valid for broadcasting.";
        rst.push_back(std::max(pad_shp1[i], pad_shp2[i]));
    }
    if (op == "dot")
        rst[rst.size()-1] = 1;
    return c10::IntArrayRef(rst);
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
    
    reversed_ = false;
}

inline std::shared_ptr<dgl::aten::CSRMatrix> AdjMatrix::GetCSC(bool inplace) {
    return this->reversed_ ? this->GetCSRMatrix(inplace) : this->GetCSCMatrix(inplace);
}

inline std::shared_ptr<dgl::aten::CSRMatrix> AdjMatrix::GetCSR(bool inplace) {
    return this->reversed_ ? this->GetCSCMatrix(inplace) : this->GetCSRMatrix(inplace);
}

inline std::shared_ptr<dgl::aten::COOMatrix> AdjMatrix::GetCOO(bool inplace) {
    return (this->reversed_ ? 
        std::make_shared<dgl::aten::COOMatrix>(dgl::aten::COOTranspose(*this->GetCOOMatrix(inplace))) 
        : this->GetCOOMatrix(inplace));
}

std::shared_ptr<dgl::aten::CSRMatrix> AdjMatrix::GetCSCMatrix(bool inplace) {
    if (inplace && !(format_ & dgl::CSC_CODE))
        LOG(FATAL) << "Format not allowed";
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
                LOG(FATAL) << "No format";
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

void AdjMatrix::SpMM(const std::string & op, const std::string & reduce,
                        dgl::NDArray ufeat, dgl::NDArray efeat,
                        dgl::NDArray out, std::vector<dgl::NDArray> out_aux) {
    dgl::SparseFormat format = this->SelectFormat(dgl::CSC_CODE);
    const auto & bcast = dgl::CalcBcastOff(op, ufeat, efeat);

    ATEN_XPU_SWITCH_CUDA(this->Context().device_type, XPU, "SpMM", {
        ATEN_ID_TYPE_SWITCH(this->DataType(), IdType, {
            ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "Feature data", {
                if (format == dgl::SparseFormat::kCSC) {
                    dgl::aten::SpMMCsr<XPU, IdType, bits>(
                        op, reduce, bcast, *this->GetCSC(),
                        ufeat, efeat, out, out_aux);
                } else if (format == dgl::SparseFormat::kCOO) {
                    dgl::aten::SpMMCoo<XPU, IdType, bits>(
                        op, reduce, bcast, *this->GetCOO(),
                        ufeat, efeat, out, out_aux);
                } else {
                    LOG(FATAL) << "SpMM only supports CSC and COO formats";
                }
            });
        });
    });
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
                    AdjMatrix::_gspmm(const std::string & op, 
                            const std::string & reduce,
                            c10::optional<torch::Tensor> ufeat, 
                            c10::optional<torch::Tensor> efeat) {
    bool use_u = (op != "copy_rhs");
    bool use_e = (op != "copy_lhs");

    if (use_u && use_e) {
        if (ufeat.value().dtype() != efeat.value().dtype())
            LOG(FATAL) << "The node features' data type doesn't match edge";
    }

    // deal with scalar feature
    bool expand_u = false, expand_e = false;
    if (use_u && ufeat.value().ndimension() == 1)
        ufeat = ufeat.value().unsqueeze(-1);
    if (use_e && efeat.value().ndimension() == 1)
        efeat = efeat.value().unsqueeze(-1);
    
    auto ctx = (use_u ? ufeat.value().device() : efeat.value().device());
    auto dtype = (use_u ? ufeat.value().dtype() : efeat.value().dtype());
    auto u_shp = (use_u ? ufeat.value().sizes() : c10::IntArrayRef({0}));
    auto e_shp = (use_e ? efeat.value().sizes() : c10::IntArrayRef({0}));

    std::vector<int64_t> v_shp_vec;
    v_shp_vec.push_back(this->dst_num());
    auto infer_rst = infer_broadcast_shape(op, u_shp.slice(1, u_shp.size()-1), e_shp.slice(1, e_shp.size()-1));
    for (int i = 0; i < infer_rst.size(); ++i)
        v_shp_vec.push_back(infer_rst[i]);
    auto v_shp = c10::IntArrayRef(v_shp_vec);

    auto v = torch::zeros(v_shp, torch::TensorOptions().dtype(dtype).device(ctx));

    bool use_cmp = (reduce == "min" || reduce == "max");
    auto idtype = get_dtype(this->DataType());

    c10::optional<torch::Tensor> arg_u = c10::nullopt, arg_e = c10::nullopt;
    if (use_cmp && use_u)
        arg_u = torch::zeros(v_shp, torch::TensorOptions().dtype(idtype).device(ctx));
    if (use_cmp && use_e)
        arg_e = torch::zeros(v_shp, torch::TensorOptions().dtype(idtype).device(ctx));
    
    auto arg_u_nd = ToNDArray(arg_u);
    auto arg_e_nd = ToNDArray(arg_e);
    auto out = ToNDArray(v);

    this->SpMM(op, reduce, ToNDArray(ufeat), ToNDArray(efeat), out, {arg_u_nd, arg_e_nd});

    v = at::fromDLPack(out.ToDLPack());
    if (!arg_u)
        arg_u = at::fromDLPack(arg_u_nd.ToDLPack());
    if (!arg_e)
        arg_e = at::fromDLPack(arg_e_nd.ToDLPack());
    
    // To deal with scalar node/edge feature
    if ((expand_u || !use_u) && (expand_e || !use_e))
        v = v.squeeze(-1);
    if (expand_u && use_cmp)
        arg_u = arg_u.value().squeeze(-1);
    if (expand_e && use_cmp)
        arg_e = arg_e.value().squeeze(-1);

    return std::make_tuple(v, arg_u, arg_e);
} 

torch::Tensor AdjMatrix::gspmm(std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat, c10::optional<torch::Tensor> efeat) {
    if (op == "sub") {
        op = "add";
        efeat = -efeat.value();
    }
    if (op == "div") {
        op = "mul";
        efeat = efeat.value().reciprocal();
    }
    auto rst = std::get<0>(this->_gspmm(op, reduce, ufeat, efeat));
    return rst;
}

torch::Tensor AdjMatrix::indegrees(c10::optional<torch::Tensor> node_ids) {
    dgl::SparseFormat fmt = this->SelectFormat(dgl::CSC_CODE);
    if (!node_ids) {
        node_ids = torch::arange(int64_t(0), this->dst_num(), 
                        torch::TensorOptions().
                        dtype(get_dtype(this->DataType())).
                        device(get_device(this->Context())));
    }
    auto vids = ToNDArray(node_ids);
    dgl::NDArray degs;
    if (fmt == dgl::SparseFormat::kCSC) {
        degs = dgl::aten::CSRGetRowNNZ(*this->GetCSC(), vids);
    } 
    else if (fmt == dgl::SparseFormat::kCOO) {
        degs = dgl::aten::COOGetRowNNZ(dgl::aten::COOTranspose(*this->GetCOO()), vids);
    }
    else
        LOG(FATAL) << "InDegree only supports CSC and COO formats";
    return at::fromDLPack(degs.ToDLPack());
}

torch::Tensor AdjMatrix::outdegrees(c10::optional<torch::Tensor> node_ids) {
    dgl::SparseFormat fmt = this->SelectFormat(dgl::CSR_CODE);
    if (!node_ids) {
        node_ids = torch::arange(int64_t(0), this->src_num(), 
                        torch::TensorOptions().
                        dtype(get_dtype(this->DataType())).
                        device(get_device(this->Context())));
    }
    auto vids = ToNDArray(node_ids);
    dgl::NDArray degs;
    if (fmt == dgl::SparseFormat::kCSR)
        degs = dgl::aten::CSRGetRowNNZ(*this->GetCSR(), vids);
    else if (fmt == dgl::SparseFormat::kCOO)
        degs = dgl::aten::COOGetRowNNZ(*this->GetCOO(), vids);
    else
        LOG(FATAL) << "OutDegree only supports CSR and COO formats";
    return at::fromDLPack(degs.ToDLPack());
}

struct GSpMM : public torch::autograd::Function<GSpMM> {
    static torch::Tensor forward(
                        torch::autograd::AutogradContext* ctx,
                        const c10::intrusive_ptr<AdjMatrix> &adj,
                        std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat,
                        c10::optional<torch::Tensor> efeat) {
        auto rst = adj->_gspmm(op, reduce, ufeat, efeat);
        bool reduce_last = need_reduce_last_dim(ufeat, efeat);
        auto ushp = (ufeat ? ufeat.value().sizes() : c10::IntArrayRef({}));
        auto eshp = (efeat ? efeat.value().sizes() : c10::IntArrayRef({}));
        auto dtype = (ufeat ? ufeat.value().dtype() : efeat.value().dtype());
        auto device = (ufeat ? ufeat.value().device() : efeat.value().device());

        ctx->saved_data["adj"] = adj;
        ctx->saved_data["op"] = op;
        ctx->saved_data["reduce"] = reduce;
        ctx->saved_data["ushp"] = ushp;
        ctx->saved_data["eshp"] = eshp;
        ctx->saved_data["dtype"] = dtype.toScalarType();
        ctx->saved_data["device"] = device;
        ctx->saved_data["reduce_last"] = reduce_last;

        bool req_grad_u = (ufeat ? ufeat.value().requires_grad() : false);
        bool req_grad_e = (efeat ? efeat.value().requires_grad() : false);

        ctx->saved_data["needs_input_grad"] = c10::List<bool>({false, false, false, req_grad_u, req_grad_e});

        auto arg_u = std::get<1>(rst);
        auto arg_e = std::get<2>(rst);

        if (!spmm_cache_ufeat(op, reduce, req_grad_u, req_grad_e))
            ufeat = c10::nullopt;
        if (!spmm_cache_efeat(op, reduce, req_grad_u, req_grad_e))
            efeat = c10::nullopt;
        if (!spmm_cache_argu(op, reduce, req_grad_u, req_grad_e))
            arg_u = c10::nullopt;
        if (!spmm_cache_arge(op, reduce, req_grad_u, req_grad_e))
            arg_e = c10::nullopt;

        ctx->saved_data["ufeat"] = ufeat;
        ctx->saved_data["efeat"] = efeat;
        ctx->saved_data["arg_u"] = arg_u;
        ctx->saved_data["arg_e"] = arg_e;
        // ctx->save_for_backward({ufeat, efeat, arg_u, arg_e});

        return std::get<0>(rst);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, 
                                                torch::autograd::variable_list grad_input) {
        // grad_input[0] = grad_input[0].contiguous();

        const auto & adj = ctx->saved_data["adj"].toCustomClass<AdjMatrix>();
        const auto & op = ctx->saved_data["op"].toStringRef();
        const auto & reduce = ctx->saved_data["reduce"].toStringRef();
        auto ushp = c10::IntArrayRef(ctx->saved_data["ushp"].toIntVector());
        auto eshp = c10::IntArrayRef(ctx->saved_data["eshp"].toIntVector());
        auto dtype = ctx->saved_data["dtype"].toScalarType();
        auto device = ctx->saved_data["device"].toDevice();
        bool reduce_last = ctx->saved_data["reduce_last"].toBool();

        auto needs_input_grad = ctx->saved_data["needs_input_grad"].toBoolList();

        auto ufeat = ctx->saved_data["ufeat"].toOptional<torch::Tensor>();
        auto efeat = ctx->saved_data["efeat"].toOptional<torch::Tensor>();
        auto arg_u = ctx->saved_data["arg_u"].toOptional<torch::Tensor>();
        auto arg_e = ctx->saved_data["arg_e"].toOptional<torch::Tensor>();

        auto du = torch::Tensor();
        auto de = torch::Tensor();

        if (op != "copy_rhs" && needs_input_grad[3]) {
            adj->toggle_reversed();
            if (reduce == "sum") {
                if (op == "mul") {
                    du = adj->gspmm("mul", "sum", grad_input[0], efeat);
                }
                else if (op == "add") {
                    du = adj->gspmm("copy_lhs", "sum", grad_input[0], c10::nullopt);
                }
                else if (op == "copy_lhs") {
                    du = adj->gspmm("copy_lhs", "sum", grad_input[0], c10::nullopt);
                }
            }
            adj->toggle_reversed();
        }
        torch::autograd::variable_list output = {torch::Tensor(), torch::Tensor(), torch::Tensor(), du, de};
        return output;
    }
};

TORCH_LIBRARY(DGL, m) {
    m.class_<AdjMatrix>("AdjMatrix")
        .def(torch::init<torch::Tensor, torch::Tensor>())
        .def("indegrees", &AdjMatrix::indegrees)
        .def("outdegrees", &AdjMatrix::outdegrees);
    m.def("GSpMM", [] (const c10::intrusive_ptr<AdjMatrix> & adj,
                        std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat,
                        c10::optional<torch::Tensor> efeat) {
        return GSpMM::apply(adj, op, reduce, ufeat, efeat);                                          
    });
}

