#include <torchdglgraph/utils.h>
#include <torch/custom_class.h>
#include <torch/script.h>

dgl::NDArray ToNDArray(at::Tensor t) {
    return dgl::NDArray::FromDLPack(at::toDLPack(t.contiguous()));
}

dgl::NDArray ToNDArray(const c10::optional<at::Tensor> &t) {
    if (t) {
        return dgl::NDArray::FromDLPack(at::toDLPack(t.value().contiguous()));
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

std::vector<int64_t> infer_broadcast_shape(const std::string &op, c10::IntArrayRef shp1, c10::IntArrayRef shp2) {
    auto pad_shp1 = shp1;
    auto pad_shp2 = shp2;
    if (op == "dot" && shp1.back() != shp2.back())
        LOG(FATAL) << "Dot operator is only available for arrays with the same size on last dimension, but got {} and {}.";

    if (op == "copy_lhs")
        return shp1.vec();
    if (op == "copy_rhs")
        return shp2.vec();

    if (shp1.size() > shp2.size()) {
        std::vector<int64_t> tmpvec(shp1.size() - shp2.size(), 1);
        for (int i = 0; i < shp2.size(); ++i)
            tmpvec.push_back(shp2[i]);
        pad_shp2 = c10::IntArrayRef(tmpvec);
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
        rst[rst.size() - 1] = 1;
    return rst;
}

torch::Tensor _reduce_grad(torch::Tensor grad, c10::IntArrayRef shape) {
    auto grad_shape = grad.sizes().slice(1, grad.sizes().size() - 1);
    auto in_shape = shape.slice(1, shape.size() - 1);
    if (grad_shape.equals(in_shape)) // no need to reduce
        return grad;

    int num_to_squeeze = grad_shape.size() - in_shape.size();
    // pad inshape
    std::vector<int64_t> tmpvec;
    for (int i = 0; i < num_to_squeeze; ++i)
        tmpvec.push_back(1);
    for (int i = 0; i < in_shape.size(); ++i)
        tmpvec.push_back(in_shape[i]);

    std::vector<int64_t> reduce_idx;
    for (int i = 0; i < tmpvec.size(); ++i)
    {
        auto tmp = grad_shape[i] - tmpvec[i];
        if (tmp != 0)
            reduce_idx.push_back(i + 1);
    }

    if (reduce_idx.size() > 0)
        grad = grad.sum(reduce_idx, true, c10::nullopt);

    std::vector<int64_t> rst_shape;
    rst_shape.push_back(-1);
    for (int i = 0; i < in_shape.size(); ++i) {
        rst_shape.push_back(in_shape[i]);
    }
    return grad.view(rst_shape);
}

std::vector<int64_t> shape_concat(c10::IntArrayRef a, c10::IntArrayRef b) {
    std::vector<int64_t> rst_vec;
    for (int i = 0; i < a.size(); ++i)
        rst_vec.push_back(a[i]);
    for (int i = 0; i < b.size(); ++i)
        rst_vec.push_back(b[i]);
    return rst_vec;
}

torch::Tensor _expand(torch::Tensor x, c10::IntArrayRef shape) {
    return x.expand(shape_concat({-1}, shape));
}

std::vector<int64_t> to_vec(c10::IntArrayRef s) {
    std::vector<int64_t> rst;
    for (size_t i = 0; i < s.size(); ++i)
        rst.push_back(s[i]);
    return rst;
}