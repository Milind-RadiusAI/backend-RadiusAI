#include "utils.h"
#include "barcode_seg_process.h"
#include "cpp_dot_product.h"
#include "hmm.h"
#include "PLCppModule.h"
BaseModel::BaseModel() {}

void ModelFactory::RegisterType() {
         registry["dot_product"] = [](std::vector<int64_t> &input_shape) -> std::shared_ptr<BaseModel> { return std::make_shared<CppDotProduct>(); };
         registry["barcode_seg_process"] = [](std::vector<int64_t> &input_shape) -> std::shared_ptr<BaseModel> { return std::make_shared<BarcodeSegProcess>(); };
         registry["hmm"] = [](std::vector<int64_t> &input_shape) -> std::shared_ptr<BaseModel> { return std::make_shared<HMMCppModule>(); };
         registry["projection_logic"] = [](std::vector<int64_t> &input_shape) -> std::shared_ptr<BaseModel> { return std::make_shared<PLCppModule>(input_shape); };
}


TRITONSERVER_Error* ModelFactory::CreateInstance(const std::string& name, std::shared_ptr<BaseModel>& model_class, std::vector<int64_t>& input_shape) {
      auto it = registry.find(name);
      if (it != registry.end())
      {
          model_class = it->second(input_shape);
          return nullptr;
      }
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "RAI custom backend version does not support this model");
            //return nullptr; 
}

void deserialize_inputs(const std::vector<float*>& dataPointers, const std::vector<std::vector<std::int64_t>>& shapes,
                        const std::vector<std::string>& data_types, std::vector<torch::Tensor>& deserializedDatasets)
{
    torch::TensorOptions options_uint = torch::TensorOptions().dtype(torch::kUInt8);
    torch::TensorOptions options_float32 = torch::TensorOptions().dtype(torch::kFloat32);
    torch::TensorOptions options_int32 = torch::TensorOptions().dtype(torch::kInt32);

    for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
            num_elements *= dim; new_shape.push_back(dim);
        }
        if (data_types[i] == "TYPE_UINT8") {

        deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape,options_uint));
        }
        else if (data_types[i] == "TYPE_INT32") {
        deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_int32));
        }
        else if (data_types[i] == "TYPE_FP32") {
        deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_float32));
        }
        else{
                continue;
        }
    }
}