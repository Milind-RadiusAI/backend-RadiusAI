#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <functional>
#include <map>
#include <memory>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class BaseModel
{
    private:
    public:
        BaseModel();
        virtual void initialize() = 0;
        virtual void execute(
            const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes,
            const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,
            std::vector<std::vector<std::int64_t>>&  output_shapes) = 0;
};

class ModelFactory
{
    private:
        std::map<std::string, std::function<std::shared_ptr<BaseModel>(std::vector<int64_t>&)>> registry;
    public:
        void RegisterType();
        TRITONSERVER_Error* CreateInstance(const std::string& name, std::shared_ptr<BaseModel>& model_class, std::vector<int64_t>& input_shape);
};

void deserialize_inputs(const std::vector<float*>& dataPointers, const std::vector<std::vector<std::int64_t>>& shapes,
                        const std::vector<std::string>& data_types, std::vector<torch::Tensor>& deserializedDatasets);

class PipelineConf {
public:
    explicit PipelineConf() {
        const char *pipeline_conf_env = getenv("PIPELINE_CONF");
        const char *client_env = getenv("ENV_CLIENT");

        std::string pipeline_conf_file(pipeline_conf_env);
        std::string client(client_env);

        std::string conf_path = "/conf/" + client + "/pipeline/" + pipeline_conf_file;

        YAML::Node pipe_conf;
        try {
            pipe_conf = YAML::LoadFile(conf_path);
        } catch (const std::exception &e) {
            std::cerr << "Error loading pipeline configuration file: " << e.what() << std::endl;
            throw;
        }
        
        // Initialize class members from the YAML config
        input_cam_sources = pipe_conf["input_cam_sources"];
        streaminfer_config = pipe_conf["streaminfer_config"];
    }

    YAML::Node input_cam_sources;
    YAML::Node streaminfer_config;
};

#endif 