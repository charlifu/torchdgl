#ifndef INCLUDE_TORCHDGLGRAPH_UTILS_H_
#define INCLUDE_TORCHDGLGRAPH_UTILS_H_

#include <dgl/aten/array_ops.h>
#include <dgl/bcast.h>
#include <dgl/aten/types.h>
#include <array/kernel_decl.h>


#include <ATen/core/TensorBody.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/DLConvertor.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>
#include <vector>
#include <memory>
#include <map>

dgl::NDArray ToNDArray(at::Tensor t); 

dgl::NDArray ToNDArray(const c10::optional<at::Tensor> & t); 

c10::Device get_device(DLContext ctx);

c10::ScalarType get_dtype(DLDataType dtype); 

std::vector<int64_t> infer_broadcast_shape(const std::string & op, c10::IntArrayRef shp1, c10::IntArrayRef shp2);

torch::Tensor _reduce_grad(torch::Tensor grad, c10::IntArrayRef shape);

std::vector<int64_t> shape_concat(c10::IntArrayRef a, c10::IntArrayRef b);

torch::Tensor _expand(torch::Tensor x, c10::IntArrayRef shape);

static std::map<std::string, int> target_mapping {
    {"u", 0},
    {"e", 1},
    {"v", 2},
    {"src", 0},
    {"edge", 1},
    {"dst", 2}
};

#endif // INCLUDE_TORCHDGLGRAPH_UTILS_H_