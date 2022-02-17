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

/*! \brief Generalized Edge_softmax op for forward */
void AdjMatrix::Edge_softmax_forward(const std::string& op,
          NDArray ufeat,
          NDArray efeat,
          NDArray out) {
  // TODO(zhejiang): add gpu op for edge_softmax
  dgl::SparseFormat format = this->SelectFormat(dgl::CSC_CODE);
  const auto& bcast = dgl::CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH(this->Context().device_type, XPU, "edge_softmax", {
    ATEN_ID_TYPE_SWITCH(this->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "edge_softmax out data", {
        dgl::aten::Edge_softmax_csr_forward<XPU, IdType, bits>(
          op, bcast, *this->GetCSC(), ufeat, efeat, out);
      });
    });
  });
}

/*! \brief Generalized Edge_softmax op for backward */
void AdjMatrix::Edge_softmax_backward(const std::string& op,
          NDArray out,
          NDArray sds,
          NDArray back_out,
          NDArray ufeat) {
  // TODO(zhejiang): add gpu op for edge_softmax
  dgl::SparseFormat format = this->SelectFormat(dgl::CSC_CODE);
  const auto& bcast = dgl::CalcBcastOff(op, ufeat, sds);

  ATEN_XPU_SWITCH(this->Context().device_type, XPU, "edge_softmax_back", {
    ATEN_ID_TYPE_SWITCH(this->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "edge_softmax out data_back", {
        dgl::aten::Edge_softmax_csr_backward<XPU, IdType, bits>(
          op, bcast, *this->GetCSC(), out, sds, back_out);
      });
    });
  });
}

torch::Tensor AdjMatrix::_edge_softmax_forward(c10::optional<torch::Tensor> e,
                                        const std::string & op) {
    bool expand = false;
    if (e.value().ndimension() == 1) {
        e = e.value().unsqueeze(-1);
        expand = true;
    }
    auto myout = torch::zeros_like(e.value());
    this->Edge_softmax_forward(op, 
                            ToNDArray(c10::nullopt),
                            ToNDArray(e),
                            ToNDArray(myout));
    if (expand) {
        myout = myout.squeeze(-1);
    }
    return myout;
}

torch::Tensor AdjMatrix::_edge_softmax_backward(c10::optional<torch::Tensor> out,
                                        c10::optional<torch::Tensor> sds) {
    std::string op = "copy_rhs";
    auto back_out = torch::zeros_like(out.value());
    this->Edge_softmax_backward(op,
                                ToNDArray(out),
                                ToNDArray(sds),
                                ToNDArray(back_out),
                                ToNDArray(c10::nullopt));
    return back_out;
}

struct EdgeSoftmax : public torch::autograd::Function<EdgeSoftmax> {
    static torch::Tensor forward(
                        torch::autograd::AutogradContext* ctx,
                        const c10::intrusive_ptr<AdjMatrix> &adj,
                        c10::optional<torch::Tensor> score,
                        std::string norm_by) {
        if (norm_by == "src")
            adj->toggle_reversed();

        auto out = torch::Tensor();
        if (score.value().is_cuda()) {
            auto score_max = std::get<0>(adj->_gspmm("copy_rhs", 
                                        "max", 
                                        c10::nullopt,
                                        score));

            score = torch::exp(adj->_gsddmm("sub", score, score_max, "e", "v"));

            auto score_sum = std::get<0>(adj->_gspmm("copy_rhs", 
                                                    "sum", 
                                                    c10::nullopt, 
                                                    score));
            out = adj->_gsddmm("div", score, score_sum, "e", "v");
        }   
        else {
            out = adj->_edge_softmax_forward(score, "copy_rhs");
        }     

        if (norm_by == "src")
            adj->toggle_reversed();
        
        ctx->saved_data["adj"] = adj;
        ctx->save_for_backward({out});

        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext*ctx,
                                                torch::autograd::variable_list grad_input) {
        const auto & adj = ctx->saved_data["adj"].toCustomClass<AdjMatrix>();
        auto out = ctx->get_saved_variables()[0];

        auto sds = out * grad_input[0];

        auto grad_score = torch::Tensor();
        if (out.is_cuda()) {
            auto accum = adj->gspmm("copy_rhs", "sum", c10::nullopt, sds);
            grad_score = sds - adj->gsddmm("mul", out, accum, "e", "v");
        }
        else {
            grad_score = adj->_edge_softmax_backward(out, sds);
        }
        return {torch::Tensor(), grad_score, torch::Tensor()};
    }
};

TORCH_LIBRARY_FRAGMENT(DGL, m) {
    m.def("EdgeSoftmax", [] (const c10::intrusive_ptr<AdjMatrix> & adj,
                        c10::optional<torch::Tensor> score,
                        std::string norm_by) {
        return EdgeSoftmax::apply(adj, score, norm_by);                                          
    });
}