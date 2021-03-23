// This header is all you need to do the C++ portions of this
// tutorial
#include <torchdglgraph/torch_dgl_graph.h>
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

TorchDGLGraph::TorchDGLGraph(at::Tensor src, at::Tensor dst) {
  int64_t num_src = at::max(src).item().toLong() + 1;
  int64_t num_dst = at::max(dst).item().toLong() + 1;
  hgptr =
      dgl::CreateFromCOO(1, num_src, num_dst, ToNDArray(src), ToNDArray(dst));
}

torch::Tensor TorchDGLGraph::in_degrees(torch::Tensor vids) {
  NDArray nd_vids = dgl::NDArray::FromDLPack(at::toDLPack(vids));
  NDArray ret_nd = hgptr->InDegrees(0, nd_vids);
  return at::fromDLPack(ret_nd.ToDLPack());
}

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
  m.class_<TorchDGLGraph>("TorchDGLGraph")
      // The following line registers the contructor of our MyStackClass
      // class that takes a single `std::vector<std::string>` argument,
      // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
      // Currently, we do not support registering overloaded
      // constructors, so for now you can only `def()` one instance of
      // `torch::init`.
      .def(torch::init<at::Tensor, at::Tensor>())
      // The next line registers a stateless (i.e. no captures) C++ lambda
      // function as a method. Note that a lambda function must take a
      // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
      // as the first argument. Other arguments can be whatever you want.
      .def("in_degrees", &TorchDGLGraph::in_degrees);
  ;
}