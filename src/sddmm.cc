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

bool sddmm_cache_X(const std::string & op, 
                  bool req_grad_X,
                  bool req_grad_Y) {
    if ((op == "mul" || op == "dot") && req_grad_Y)
        return true;
    return false;
}

bool sddmm_cache_Y(const std::string & op,
                  bool req_grad_X,
                  bool req_grad_Y) {
    if ((op == "mul" || op == "dot") && req_grad_X)
        return true;
    return false;
}

void AdjMatrix::SDDMM(const std::string & op,
                    dgl::NDArray lhs,
                    dgl::NDArray rhs,
                    dgl::NDArray out,
                    int lhs_target, int rhs_target) {
  dgl::SparseFormat format = this->SelectFormat(dgl::COO_CODE);
  const auto &bcast = dgl::CalcBcastOff(op, lhs, rhs);

  ATEN_XPU_SWITCH_CUDA(this->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(this->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "Feature data", {
        if (format == dgl::SparseFormat::kCSR) {
          dgl::aten::SDDMMCsr<XPU, IdType, bits>(
              op, bcast, *this->GetCSR(),
              lhs, rhs, out, lhs_target, rhs_target);
        } else if (format == dgl::SparseFormat::kCOO) {
          dgl::aten::SDDMMCoo<XPU, IdType, bits>(
              op, bcast, *this->GetCOO(),
              lhs, rhs, out, lhs_target, rhs_target);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO formats";
        }
      });
    });
  });
}

torch::Tensor AdjMatrix::gsddmm(std::string op, c10::optional<torch::Tensor> lhs_data,
                            c10::optional<torch::Tensor> rhs_data,
                            std::string lhs_target, std::string rhs_target) {
    if (op == "sub") {
        op = "add";
        rhs_data = -rhs_data.value();
    }

    if (op == "div") {
        op = "mul";
        rhs_data = rhs_data.value().reciprocal();
    }

    return this->_gsddmm(op, lhs_data, rhs_data, lhs_target, rhs_target);
}

torch::Tensor AdjMatrix::_gsddmm(const std::string & op, 
                                c10::optional<torch::Tensor> lhs,
                                c10::optional<torch::Tensor> rhs,
                                std::string lhs_target,
                                std::string rhs_target) {
    bool use_lhs = (op != "copy_rhs");
    bool use_rhs = (op != "copy_lhs");

    if (use_lhs && use_rhs && lhs.value().dtype() != rhs.value().dtype()) {
        LOG(FATAL) << "The operands data type don't match.";
    }

    bool expand_lhs = false, expand_rhs = false;
    if (use_lhs && lhs.value().ndimension() == 1) {
        lhs = lhs.value().unsqueeze(-1);
        expand_lhs = true;
    }
    if (use_rhs && rhs.value().ndimension() == 1) {
        rhs = rhs.value().unsqueeze(-1);
        expand_rhs = true;
    }

    auto ctx = (use_lhs ? lhs.value().device() : rhs.value().device());
    auto dtype = (use_lhs ? lhs.value().dtype() : rhs.value().dtype());
    auto lhs_shp = (use_lhs ? lhs.value().sizes() : c10::IntArrayRef({0}));
    auto rhs_shp = (use_rhs ? rhs.value().sizes() : c10::IntArrayRef({0}));

    std::vector<int64_t> out_shp_vec;
    out_shp_vec.push_back(this->edge_num());
    auto infer_rst = infer_broadcast_shape(op, 
                                    lhs_shp.slice(1, lhs_shp.size()-1), 
                                    rhs_shp.slice(1, rhs_shp.size()-1));
    for (int i = 0; i < infer_rst.size(); ++i)
        out_shp_vec.push_back(infer_rst[i]);
    auto out_shp = c10::IntArrayRef(out_shp_vec);

    auto out = torch::zeros(out_shp, 
                          torch::TensorOptions().dtype(dtype).device(ctx));

    this->SDDMM(op, 
                ToNDArray(lhs),
                ToNDArray(rhs),
                ToNDArray(out),
                target_mapping[lhs_target],
                target_mapping[rhs_target]);

    if ((expand_lhs || !use_lhs) && (expand_rhs || !use_rhs))
        out = out.squeeze(-1);

    return out;
}


struct GSDDMM : public torch::autograd::Function<GSDDMM> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                    const c10::intrusive_ptr<AdjMatrix> &adj,
                                    std::string op,
                                    c10::optional<torch::Tensor> X,
                                    c10::optional<torch::Tensor> Y,
                                    std::string lhs_target, std::string rhs_target) {
        auto out = adj->_gsddmm(op, X, Y, lhs_target, rhs_target);
        auto X_shape = (X ? X.value().sizes() : c10::IntArrayRef({}));
        auto Y_shape = (Y ? Y.value().sizes() : c10::IntArrayRef({}));

        ctx->saved_data["adj"] = adj;
        ctx->saved_data["op"] = op;
        ctx->saved_data["lhs_target"] = lhs_target;
        ctx->saved_data["rhs_target"] = rhs_target;
        ctx->saved_data["X_shape"] = to_vec(X_shape);
        ctx->saved_data["Y_shape"] = to_vec(Y_shape);

        bool req_grad_X = (X ? X.value().requires_grad() : false);
        bool req_grad_Y = (Y ? Y.value().requires_grad() : false);

        ctx->saved_data["needs_input_grad"] = c10::List<bool>({false, 
                                                          false, 
                                                          req_grad_X, 
                                                          req_grad_Y,
                                                          false,
                                                          false});


        if (!sddmm_cache_X(op, req_grad_X, req_grad_Y))
            X = c10::nullopt;
        if (!sddmm_cache_Y(op, req_grad_X, req_grad_Y))
            Y = c10::nullopt;
        
        ctx->saved_data["X"] = X;
        ctx->saved_data["Y"] = Y;
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, 
                                            torch::autograd::variable_list grad_input) {
        
        const auto & adj = ctx->saved_data["adj"].toCustomClass<AdjMatrix>();
        const auto & op = ctx->saved_data["op"].toStringRef();
        const auto & lhs_target = ctx->saved_data["lhs_target"].toStringRef();
        const auto & rhs_target = ctx->saved_data["rhs_target"].toStringRef();

        auto X_vec = ctx->saved_data["X_shape"].toIntVector();
        auto X_shape = c10::IntArrayRef(X_vec);
        auto Y_vec = ctx->saved_data["Y_shape"].toIntVector();
        auto Y_shape = c10::IntArrayRef(Y_vec);
        
        auto needs_input_grad = ctx->saved_data["needs_input_grad"].toBoolList();

        auto X = ctx->saved_data["X"].toOptional<torch::Tensor>();
        auto Y = ctx->saved_data["Y"].toOptional<torch::Tensor>();

        auto dX = torch::Tensor();
        auto dY = torch::Tensor();

        if (op != "copy_rhs" && needs_input_grad[2]) {
            if (lhs_target == "u" || lhs_target == "v") {
                if (lhs_target == "v") 
                    adj->toggle_reversed();
                if (op == "add" || op == "copy_lhs")
                    dX = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0]);
                else { // mul, dot
                    if (rhs_target == lhs_target)
                        dX = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0]) * Y.value();
                    else if (rhs_target == "e")
                        dX = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0] * Y.value());
                    else
                        dX = adj->gspmm("mul", "sum", Y.value(), grad_input[0]);
                }
                if (lhs_target == "v") 
                    adj->toggle_reversed();
            }
            else { // lhs_target == "e"
                if (op == "add" || op == "copy_lhs")
                    dX = grad_input[0];
                else
                    dX = adj->gsddmm("mul", grad_input[0], Y, "e", rhs_target);
            }
            dX = _reduce_grad(dX, X_shape);
        }

        if (op != "copy_lhs" && needs_input_grad[3]) {
            if (rhs_target == "u" || rhs_target == "v") {
                if (rhs_target == "v")
                    adj->toggle_reversed();
                
                if (op == "add" && op == "copy_rhs")
                    dY = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0]);
                else { // mul, dot
                    if (lhs_target == rhs_target)
                        dY = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0]) * X.value();
                    else if (lhs_target == "e")
                        dY = adj->gspmm("copy_rhs", "sum", c10::nullopt, grad_input[0] * X.value());
                    else // rhs_target == !lhs_target
                        dY = adj->gspmm("mul", "sum", X, grad_input[0]);
                }

                if (rhs_target == "v")
                    adj->toggle_reversed();
            }
            else {
                if (op == "add" || op == "copy_rhs")
                    dY = grad_input[0];
                else // mul, dot
                    dY = adj->gsddmm("mul", grad_input[0], X, "e", lhs_target);
            }
            dY = _reduce_grad(dY, Y_shape);
        }

        torch::autograd::variable_list output = {torch::Tensor(), 
                                                torch::Tensor(), 
                                                dX,
                                                dY,
                                                torch::Tensor(),
                                                torch::Tensor()};
        return output;
    }
};

TORCH_LIBRARY_FRAGMENT(DGL, m) {
    m.def("GSDDMM", [] (const c10::intrusive_ptr<AdjMatrix> & adj,
                        std::string op,
                        c10::optional<torch::Tensor> X,
                        c10::optional<torch::Tensor> Y,
                        std::string lhs_target, std::string rhs_target) {
        return GSDDMM::apply(adj, op, X, Y, lhs_target, rhs_target);                                          
    });
}