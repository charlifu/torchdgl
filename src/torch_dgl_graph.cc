// This header is all you need to do the C++ portions of this
// tutorial

#include "dgl/aten/array_ops.h"
#include <torchdglgraph/torch_dgl_graph.h>

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

using dgl::NDArray;

NDArray ToNDArray(at::Tensor t) {
  return dgl::NDArray::FromDLPack(at::toDLPack(t));
}
NDArray ToNDArray(const c10::optional<at::Tensor>& t) {
  if (t) {
    return dgl::NDArray::FromDLPack(at::toDLPack(t.value()));
  } else {
    return dgl::aten::NullArray();
  }
}

TorchDGLGraph::TorchDGLGraph(at::Tensor src, at::Tensor dst) {
  int64_t num_src = at::max(src).item().toLong() + 1;
  int64_t num_dst = at::max(dst).item().toLong() + 1;
  hgptr =
      dgl::CreateFromCOO(1, num_src, num_dst, ToNDArray(src), ToNDArray(dst));
}

torch::Tensor TorchDGLGraph::InDegrees(torch::Tensor vids) {
  // vids.sizes();
  NDArray nd_vids = dgl::NDArray::FromDLPack(at::toDLPack(vids));
  NDArray ret_nd = hgptr->InDegrees(0, nd_vids);
  return at::fromDLPack(ret_nd.ToDLPack());
}
int64_t TorchDGLGraph::NumETypes() { return hgptr->NumEdgeTypes(); }
std::array<int64_t, 2> TorchDGLMetaGraph::FindEdge(int64_t etypes) {
  auto pair = metagraph->FindEdge(etypes);
  return {static_cast<long>(pair.first), static_cast<long>(pair.second)};
}

c10::intrusive_ptr<TorchDGLMetaGraph> TorchDGLGraph::GetMetaGraph() {
  return c10::make_intrusive<TorchDGLMetaGraph>(hgptr->meta_graph());
}
int64_t TorchDGLGraph::NumVertexs(int64_t etypes) {
  return hgptr->NumVertices(etypes);
};
std::string TorchDGLGraph::DataType() {
  auto dtype = hgptr->DataType();
  if (dtype.bits == 32) {
    return "int32";
  } else {
    return "int64";
  }
};
int64_t TorchDGLGraph::NumEdges(int64_t etypes) {
  return hgptr->NumEdges(etypes);
}
// std::array<int64_t, 2> TorchDGLMetaGraph::FindEdges(int64_t etypes){

// }

// Notice a few things:
// - We pass the class to be registered as a template parameter to
//   `torch::class_`. In this instance, we've passed the
//   specialization of the MyStackClass class ``MyStackClass<std::string>``.
//   In general, you cannot register a non-specialized template
//   class. For non-templated classes, you can just pass the
//   class name directly as the template parameter.
// - The arguments passed to the constructor make up the "qualified name"
//   of the class. In this case, the registered class will appear in
//   Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
//   the first argument the "namespace" and the second argument the
//   actual class name.

TORCH_LIBRARY(my_classes, m) {
  m.class_<TorchDGLMetaGraph>("TorchDGLMetaGraph")
      .def("find_edge", &TorchDGLMetaGraph::FindEdge);
  m.class_<TorchDGLGraph>("TorchDGLGraph")
      // The following line registers the contructor of our MyStackClass
      // class that takes a single `std::vector<std::string>` argument,
      // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
      // Currently, we do not support registering overloaded
      // constructors, so for now you can only `def()` one instance of
      // `torch::init`.
      // .def_static(std::string name, Func func)
      .def(torch::init<at::Tensor, at::Tensor>())
      .def_property("metagraph", &TorchDGLGraph::GetMetaGraph)
      .def_property("dtype", &TorchDGLGraph::DataType)
      .def("in_degrees", &TorchDGLGraph::InDegrees)
      .def("number_of_etypes", &TorchDGLGraph::NumETypes)
      .def("number_of_nodes", &TorchDGLGraph::NumVertexs)
      .def("number_of_edges", &TorchDGLGraph::NumEdges);
      // m.def("my_classes::_CAPI_DGLKernelSpMM(__torch__.torch.classes.my_classes.TorchDGLGraph _0, str _1, str _2, Tensor? _3, Tensor? _4, Tensor? _5, Tensor? _6, Tensor? _7) -> ()",
      m.def("_CAPI_DGLKernelSpMM",
        [](const c10::intrusive_ptr<TorchDGLGraph> &hg, std::string op,
           std::string reduce_op, c10::optional<torch::Tensor> u,
           c10::optional<torch::Tensor> e, c10::optional<torch::Tensor> v, c10::optional<torch::Tensor> arg_u,
           c10::optional<torch::Tensor> arg_e) {
          NDArray U =ToNDArray(u);
          NDArray E = ToNDArray(e);
          NDArray V = ToNDArray(v);
          NDArray ArgU = ToNDArray(arg_u);
          NDArray ArgE = ToNDArray(arg_e);
          dgl::aten::SpMM(op, reduce_op, hg->hgptr, U, E, V, {ArgU, ArgE});
        });
  // .def("metagraph")
}
