#include "cpp_dot_product.h"
using namespace torch::indexing;

CppDotProduct::CppDotProduct() {}


void CppDotProduct::initialize() {}


void CppDotProduct::execute(
    const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,
    std::vector<std::vector<std::int64_t>>&  output_shapes
) 
{
    std::vector<torch::Tensor> all_inputs;
    deserialize_inputs(dataPointers,shapes,data_types,all_inputs);
    torch::Tensor result = torch::mm(all_inputs[0], all_inputs[1]);
    torch::IntArrayRef sizes = result.sizes();
    std::vector<int64_t> shape_(sizes.begin(), sizes.end());
    output_shapes.push_back(shape_);
    float* heapTensorData = new float[result.numel()];
    std::memcpy(heapTensorData, result.data_ptr<float>(), result.numel() * sizeof(float));
    outputPointers.push_back(heapTensorData);
}