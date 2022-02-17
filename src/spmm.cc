#include <torchdglgraph/adjmatrix.h>
#include <torchdglgraph/utils.h>
#include <dgl/aten/array_ops.h>
#include <dgl/bcast.h>
#include <array/kernel_decl.h>


#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/custom_class.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/op_registration/infer_schema.h>

#include <string>
#include <vector>

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
    auto infer_rst = infer_broadcast_shape(op, 
                                    u_shp.slice(1, u_shp.size()-1), 
                                    e_shp.slice(1, e_shp.size()-1));
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
    // auto out = ToNDArray(v);

    this->SpMM(op, reduce, ToNDArray(ufeat), ToNDArray(efeat), ToNDArray(v), {arg_u_nd, arg_e_nd});

    // v = at::fromDLPack(out.ToDLPack());
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
        ctx->saved_data["ushp"] = to_vec(ushp);
        ctx->saved_data["eshp"] = to_vec(eshp);
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

        auto uvec = ctx->saved_data["ushp"].toIntVector();
        auto ushp = c10::IntArrayRef(uvec);
        auto evec = ctx->saved_data["eshp"].toIntVector();
        auto eshp = c10::IntArrayRef(evec);

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
            else { // max/min
                auto dZ_dim = grad_input[0].sizes().size();
                du = torch::zeros(shape_concat(ushp.slice(1), grad_input[0].sizes().slice(1,dZ_dim-1)),
                                    torch::TensorOptions()
                                    .dtype(dtype)
                                    .device(device));
                if (op == "mul") {
                    auto grad = _expand(efeat.value(), grad_input[0].sizes().slice(1,dZ_dim-1))
                            .gather(0, arg_e.value().toType(torch::kInt64)) * grad_input[0];
                    du.scatter_add_(0, arg_u.value().toType(torch::kInt64), grad);
                }
                else if (op == "add" || op == "copy_lhs") {
                    du.scatter_add_(0, arg_u.value().toType(torch::kInt64), grad_input[0]);
                }
            }
            du = _reduce_grad(du, ushp);
            adj->toggle_reversed();
        }

        if (op != "copy_lhs" && needs_input_grad[4]) {
            if (reduce == "sum") {
                if (op == "mul" && reduce_last) {
                    de = adj->gsddmm("dot", ufeat, grad_input[0]);
                }
                else if (op == "mul") {
                    de = adj->gsddmm("mul", ufeat, grad_input[0]);
                }
                else if (op == "add" || op == "copy_rhs") {
                    de = adj->gsddmm("copy_rhs", ufeat, grad_input[0]);
                }
            }
            else { // max/min
                auto dZ_dim = grad_input[0].sizes().size();
                de = torch::zeros(shape_concat(eshp.slice(1), grad_input[0].sizes().slice(1, dZ_dim-1)),
                                    torch::TensorOptions()
                                    .dtype(dtype)
                                    .device(device));
                if (op == "mul") {
                    auto grad = _expand(ufeat.value(), grad_input[0].sizes().slice(1, dZ_dim-1))
                                .gather(0, arg_e.value().toType(torch::kInt64)) * grad_input[0];
                    de.scatter_add(0, arg_e.value().toType(torch::kInt64), grad);
                }
                else if (op == "add" || op == "copy_rhs") {
                    de.scatter_add_(0, arg_e.value().toType(torch::kInt64), grad_input[0]);
                }
            }
            de = _reduce_grad(de, eshp);
        }
        torch::autograd::variable_list output = {torch::Tensor(),
                                                torch::Tensor(),
                                                torch::Tensor(),
                                                du,
                                                de};
        return output;
    }
};

TORCH_LIBRARY_FRAGMENT(DGL, m) {
    m.def("GSpMM", [] (const c10::intrusive_ptr<AdjMatrix> & adj,
                        std::string op, std::string reduce,
                        c10::optional<torch::Tensor> ufeat,
                        c10::optional<torch::Tensor> efeat) {
        return GSpMM::apply(adj, op, reduce, ufeat, efeat);                                          
    });
}